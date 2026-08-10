/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.catalyst.expressions

import java.util.Locale

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.Cast._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.util.{ArrayData, CollationFactory, MapData}
import org.apache.spark.sql.errors.QueryCompilationErrors
import org.apache.spark.sql.types._
import org.apache.spark.types.variant.{Variant, VariantUtil}
import org.apache.spark.unsafe.types.{BinaryView, CalendarInterval, TimestampNanosVal, UTF8String, VariantVal}

private[expressions] object Xxh3HashCommon {
  // 64-bit golden-ratio constant for position mixing.
  final val POSITION_MIX: Long = 0x9E3779B97F4A7C15L
  private final val MAX_CODEGEN_STRUCT_FIELDS = 64

  // Returns seed perturbed by position p (1-indexed); identity at p == 1.
  @inline def mixPos(seed: Long, p: Int): Long =
    if (p > 1) seed ^ (p.toLong * POSITION_MIX) else seed

  @inline def mixNullRun64(seed: Long, nulls: Int, startPos: Int): Long =
    if (nulls > 0) XXH3.hashLong64(nulls.toLong, seed ^ (-startPos.toLong * POSITION_MIX))
    else seed

  @inline def mixNullRun128Into(
      seed: Long,
      nulls: Int,
      startPos: Int,
      output: Array[Byte]): Long =
    XXH3.hashLong128FoldedInto(
      nulls.toLong, seed ^ (-startPos.toLong * POSITION_MIX), output)

  def useInterpretedStructHash(fields: Array[StructField]): Boolean = {
    def fieldCount(dataType: DataType): Int = dataType match {
      case StructType(nested) =>
        nested.length + nested.iterator.map(field => fieldCount(field.dataType)).sum
      case ArrayType(elementType, _) => fieldCount(elementType)
      case MapType(keyType, valueType, _) => fieldCount(keyType) + fieldCount(valueType)
      case udt: UserDefinedType[_] => fieldCount(udt.sqlType)
      case _ => 0
    }
    fields.length + fields.iterator.map(field => fieldCount(field.dataType)).sum >
      MAX_CODEGEN_STRUCT_FIELDS
  }

  def supports(dataType: DataType): Boolean = dataType match {
    case NullType | BooleanType | ByteType | ShortType | IntegerType | LongType |
        FloatType | DoubleType | DateType | TimestampType | TimestampNTZType |
        CalendarIntervalType | BinaryType => true
    case _: StringType | _: DecimalType | _: TimeType | _: DayTimeIntervalType |
        _: YearMonthIntervalType | _: TimestampNTZNanosType | _: TimestampLTZNanosType |
        _: GeometryType | _: GeographyType | _: VariantType => true
    case ArrayType(elementType, _) => supports(elementType)
    case MapType(keyType, valueType, _) => supports(keyType) && supports(valueType)
    case StructType(fields) => fields.forall(field => supports(field.dataType))
    case udt: UserDefinedType[_] => supports(udt.sqlType)
    case _ => false
  }

}
object Xxh3LongHashFunction extends Serializable {
  import Xxh3HashCommon._

  def hash(
      value: Any,
      dataType: DataType,
      seed: Long,
      isCollationAware: Boolean = false,
      legacyCollationAwareHashing: Boolean = false): Long = {
    val physicalType = dataType match {
      case udt: UserDefinedType[_] => udt.sqlType
      case other => other
    }
    value match {
      case b: Boolean => XXH3.hashInt64(if (b) 1 else 0, seed)
      case b: Byte => XXH3.hashInt64(b, seed)
      case s: Short => XXH3.hashInt64(s, seed)
      case i: Int => XXH3.hashInt64(i, seed)
      case l: Long => XXH3.hashLong64(l, seed)

      case f: Float if f == -0.0f => XXH3.hashInt64(0, seed)
      case f: Float => XXH3.hashInt64(java.lang.Float.floatToIntBits(f), seed)

      case d: Double if d == -0.0d => XXH3.hashLong64(0L, seed)
      case d: Double => XXH3.hashLong64(java.lang.Double.doubleToLongBits(d), seed)

      case d: Decimal =>
        val precision = physicalType.asInstanceOf[DecimalType].precision
        if (precision <= Decimal.MAX_LONG_DIGITS) {
          XXH3.hashLong64(d.toUnscaledLong, seed)
        } else {
          val bytes = d.toJavaBigDecimal.unscaledValue().toByteArray
          XXH3.hashBytes64(bytes, 0, bytes.length, seed)
        }

      case c: CalendarInterval =>
        val s1 = XXH3.hashLong64(c.microseconds, seed)
        val s2 = XXH3.hashInt64(c.days, s1)
        XXH3.hashInt64(c.months, s2)

      case t: TimestampNanosVal =>
        val s1 = XXH3.hashLong64(t.epochMicros, seed)
        XXH3.hashInt64(t.nanosWithinMicro, s1)

      case a: Array[Byte] =>
        XXH3.hashBytes64(a, 0, a.length, seed)

      case v: BinaryView =>
        XXH3.hashUnsafeBytes64(v.getBaseObject, v.getBaseOffset, v.numBytes, seed)

      case s: UTF8String =>
        hashString(s, physicalType.asInstanceOf[StringType], seed,
          isCollationAware, legacyCollationAwareHashing)

      case array: ArrayData =>
        val elementType = physicalType.asInstanceOf[ArrayType].elementType
        val n = array.numElements()
        var s = XXH3.hashInt64(n, seed)
        var pos = 1
        var i = 0
        while (i < n) {
          if (!array.isNullAt(i)) {
            s = hash(array.get(i, elementType), elementType, mixPos(s, pos),
              isCollationAware, legacyCollationAwareHashing)
          }
          pos += 1
          i += 1
        }
        s

      case map: MapData =>
        val mt = physicalType.asInstanceOf[MapType]
        val (kt, vt) = mt.keyType -> mt.valueType
        val n = map.numElements()
        val keys = map.keyArray()
        val values = map.valueArray()
        var accum = 0L
        var i = 0
        while (i < n) {
          val k = keys.get(i, kt)
          val kh = hash(k, kt, 0L, isCollationAware, legacyCollationAwareHashing)
          val v = if (values.isNullAt(i)) null else values.get(i, vt)
          accum += (if (v != null) {
            hash(v, vt, kh, isCollationAware, legacyCollationAwareHashing)
          } else {
            kh
          })
          i += 1
        }
        XXH3.hashLong64(accum, seed ^ (n.toLong * POSITION_MIX))

      case row: InternalRow =>
        val types = physicalType.asInstanceOf[StructType].map(_.dataType).toArray
        val len = row.numFields
        var s = XXH3.hashInt64(len, seed)
        var pos = 1
        var i = 0
        while (i < len) {
          if (!row.isNullAt(i)) {
            s = hash(row.get(i, types(i)), types(i), mixPos(s, pos),
              isCollationAware, legacyCollationAwareHashing)
          }
          pos += 1
          i += 1
        }
        s

      case v: VariantVal =>
        hashVariant(new Variant(v.getValue(), v.getMetadata()), seed)

      case _ =>
        throw new IllegalArgumentException(s"Unsupported XXH3 input type: $dataType")
    }
  }

  private def hashString(
      s: UTF8String,
      stringType: StringType,
      seed: Long,
      isCollationAware: Boolean,
      legacyCollationAwareHashing: Boolean): Long = {
    if (stringType.supportsBinaryEquality) {
      XXH3.hashUnsafeBytes64(s.getBaseObject, s.getBaseOffset, s.numBytes, seed)
    } else if (isCollationAware) {
      val key = CollationFactory.fetchCollation(stringType.collationId)
        .sortKeyFunction.apply(s).asInstanceOf[Array[Byte]]
      XXH3.hashBytes64(key, 0, key.length, seed)
    } else if (legacyCollationAwareHashing) {
      val collation = CollationFactory.fetchCollation(stringType.collationId)
      val stringHash = if (collation.isUtf8BinaryType || collation.isUtf8LcaseType) {
        UTF8String.fromBytes(collation.sortKeyFunction.apply(s)).hashCode
      } else if (collation.supportsSpaceTrimming) {
        collation.getCollator.getCollationKey(s.trimRight.toValidString).hashCode
      } else {
        collation.getCollator.getCollationKey(s.toValidString).hashCode
      }
      XXH3.hashLong64(stringHash, seed)
    } else {
      XXH3.hashUnsafeBytes64(s.getBaseObject, s.getBaseOffset, s.numBytes, seed)
    }
  }

  private def hashVariant(variant: Variant, seed: Long): Long = {
    import Xxh3HashCommon._
    variant.getType() match {
      case VariantUtil.Type.NULL => XXH3.hashLong64(0L, seed)
      case VariantUtil.Type.BOOLEAN => XXH3.hashInt64(if (variant.getBoolean()) 1 else 0, seed)
      case VariantUtil.Type.LONG => XXH3.hashLong64(variant.getLong(), seed)
      case VariantUtil.Type.DATE => XXH3.hashInt64(variant.getLong().toInt, seed)
      case VariantUtil.Type.TIMESTAMP | VariantUtil.Type.TIMESTAMP_NTZ =>
        XXH3.hashLong64(variant.getLong(), seed)

      case VariantUtil.Type.DOUBLE =>
        val d = variant.getDouble()
        if (d == -0.0d) XXH3.hashLong64(0L, seed)
        else XXH3.hashLong64(java.lang.Double.doubleToLongBits(d), seed)

      case VariantUtil.Type.FLOAT =>
        val f = variant.getFloat()
        if (f == -0.0f) XXH3.hashInt64(0, seed)
        else XXH3.hashInt64(java.lang.Float.floatToIntBits(f), seed)

      case VariantUtil.Type.DECIMAL =>
        val bytes = variant.getDecimal().unscaledValue().toByteArray
        XXH3.hashBytes64(bytes, 0, bytes.length, seed)

      case VariantUtil.Type.STRING =>
        val s = UTF8String.fromString(variant.getString())
        XXH3.hashUnsafeBytes64(s.getBaseObject, s.getBaseOffset, s.numBytes, seed)

      case VariantUtil.Type.BINARY =>
        val b = variant.getBinary()
        XXH3.hashBytes64(b, 0, b.length, seed)

      case VariantUtil.Type.UUID =>
        val uuid = variant.getUuid()
        val s1 = XXH3.hashLong64(uuid.getMostSignificantBits, seed)
        XXH3.hashLong64(uuid.getLeastSignificantBits, mixPos(s1, 2))

      case VariantUtil.Type.OBJECT =>
        val n = variant.objectSize()
        var accum = 0L
        var i = 0
        while (i < n) {
          val field = variant.getFieldAtIndex(i)
          val kb = field.key.getBytes(java.nio.charset.StandardCharsets.UTF_8)
          val kh = XXH3.hashBytes64(kb, 0, kb.length, 0L)
          accum += hashVariant(field.value, kh)
          i += 1
        }
        XXH3.hashLong64(accum, seed ^ (n.toLong * POSITION_MIX))

      case VariantUtil.Type.ARRAY =>
        val n = variant.arraySize()
        var s = XXH3.hashInt64(n, seed)
        var pos = 1
        var i = 0
        while (i < n) {
          s = hashVariant(variant.getElementAtIndex(i), mixPos(s, pos))
          pos += 1
          i += 1
        }
        s
    }
  }
}
object Xxh3Bytes128HashFunction extends Serializable {
  import Xxh3HashCommon._

  def hash(value: Any, dataType: DataType, seed: Long): Array[Byte] = {
    val output = new Array[Byte](16)
    hashInto(value, dataType, seed, output)
    output
  }

  def hashInto(
      value: Any,
      dataType: DataType,
      seed: Long,
      output: Array[Byte]): Long = {
    val physicalType = dataType match {
      case udt: UserDefinedType[_] => udt.sqlType
      case other => other
    }
    value match {
    case b: Boolean => XXH3.hashInt128FoldedInto(if (b) 1 else 0, seed, output)
    case b: Byte => XXH3.hashInt128FoldedInto(b, seed, output)
    case s: Short => XXH3.hashInt128FoldedInto(s, seed, output)
    case i: Int => XXH3.hashInt128FoldedInto(i, seed, output)
    case l: Long => XXH3.hashLong128FoldedInto(l, seed, output)

    case f: Float if f == -0.0f => XXH3.hashInt128FoldedInto(0, seed, output)
    case f: Float =>
      XXH3.hashInt128FoldedInto(java.lang.Float.floatToIntBits(f), seed, output)

    case d: Double if d == -0.0d => XXH3.hashLong128FoldedInto(0L, seed, output)
    case d: Double =>
      XXH3.hashLong128FoldedInto(java.lang.Double.doubleToLongBits(d), seed, output)

    case d: Decimal =>
      val precision = physicalType.asInstanceOf[DecimalType].precision
      if (precision <= Decimal.MAX_LONG_DIGITS) {
        XXH3.hashLong128FoldedInto(d.toUnscaledLong, seed, output)
      } else {
        val bytes = d.toJavaBigDecimal.unscaledValue().toByteArray
        XXH3.hashBytes128FoldedInto(bytes, 0, bytes.length, seed, output)
      }

    case c: CalendarInterval =>
      val s1 = XXH3.hashLong128FoldedInto(c.microseconds, seed, output)
      val s2 = XXH3.hashInt128FoldedInto(c.days, s1, output)
      XXH3.hashInt128FoldedInto(c.months, s2, output)

    case t: TimestampNanosVal =>
      val s1 = XXH3.hashLong128FoldedInto(t.epochMicros, seed, output)
      XXH3.hashInt128FoldedInto(t.nanosWithinMicro, s1, output)

    case a: Array[Byte] =>
      XXH3.hashBytes128FoldedInto(a, 0, a.length, seed, output)

    case v: BinaryView =>
      XXH3.hashUnsafeBytes128FoldedInto(
        v.getBaseObject, v.getBaseOffset, v.numBytes, seed, output)

    case s: UTF8String =>
      XXH3.hashUnsafeBytes128FoldedInto(
        s.getBaseObject, s.getBaseOffset, s.numBytes, seed, output)

    case array: ArrayData =>
      val elementType = physicalType.asInstanceOf[ArrayType].elementType
      val n = array.numElements()
      var s = XXH3.hashInt128FoldedInto(n, seed, output)
      var pos = 1
      var i = 0
      while (i < n) {
        if (!array.isNullAt(i)) {
          s = hashInto(array.get(i, elementType), elementType, mixPos(s, pos), output)
        }
        pos += 1
        i += 1
      }
      s

    case map: MapData =>
      val mt = physicalType.asInstanceOf[MapType]
      val (kt, vt) = mt.keyType -> mt.valueType
      val n = map.numElements()
      val keys = map.keyArray()
      val values = map.valueArray()
      var accum = 0L
      var i = 0
      while (i < n) {
        val k = keys.get(i, kt)
        val kh = hashInto(k, kt, 0L, output)
        val v = if (values.isNullAt(i)) null else values.get(i, vt)
        accum += (if (v != null) hashInto(v, vt, kh, output) else kh)
        i += 1
      }
      XXH3.hashLong128FoldedInto(accum, seed ^ (n.toLong * POSITION_MIX), output)

    case row: InternalRow =>
      val types = physicalType.asInstanceOf[StructType].map(_.dataType).toArray
      val len = row.numFields
      var s = XXH3.hashInt128FoldedInto(len, seed, output)
      var pos = 1
      var i = 0
      while (i < len) {
        if (!row.isNullAt(i)) {
          s = hashInto(row.get(i, types(i)), types(i), mixPos(s, pos), output)
        }
        pos += 1
        i += 1
      }
      s

    case v: VariantVal =>
      hashVariantInto(new Variant(v.getValue(), v.getMetadata()), seed, output)

    case _ =>
      throw new IllegalArgumentException(s"Unsupported XXH3 input type: $dataType")
    }
  }

  private def hashVariantInto(variant: Variant, seed: Long, output: Array[Byte]): Long = {
    variant.getType() match {
      case VariantUtil.Type.NULL => XXH3.hashLong128FoldedInto(0L, seed, output)
      case VariantUtil.Type.BOOLEAN =>
        XXH3.hashInt128FoldedInto(if (variant.getBoolean()) 1 else 0, seed, output)
      case VariantUtil.Type.LONG => XXH3.hashLong128FoldedInto(variant.getLong(), seed, output)
      case VariantUtil.Type.DATE =>
        XXH3.hashInt128FoldedInto(variant.getLong().toInt, seed, output)
      case VariantUtil.Type.TIMESTAMP | VariantUtil.Type.TIMESTAMP_NTZ =>
        XXH3.hashLong128FoldedInto(variant.getLong(), seed, output)

      case VariantUtil.Type.DOUBLE =>
        val d = variant.getDouble()
        if (d == -0.0d) XXH3.hashLong128FoldedInto(0L, seed, output)
        else XXH3.hashLong128FoldedInto(java.lang.Double.doubleToLongBits(d), seed, output)

      case VariantUtil.Type.FLOAT =>
        val f = variant.getFloat()
        if (f == -0.0f) XXH3.hashInt128FoldedInto(0, seed, output)
        else XXH3.hashInt128FoldedInto(java.lang.Float.floatToIntBits(f), seed, output)

      case VariantUtil.Type.DECIMAL =>
        val bytes = variant.getDecimal().unscaledValue().toByteArray
        XXH3.hashBytes128FoldedInto(bytes, 0, bytes.length, seed, output)

      case VariantUtil.Type.STRING =>
        val s = UTF8String.fromString(variant.getString())
        XXH3.hashUnsafeBytes128FoldedInto(
          s.getBaseObject, s.getBaseOffset, s.numBytes, seed, output)

      case VariantUtil.Type.BINARY =>
        val b = variant.getBinary()
        XXH3.hashBytes128FoldedInto(b, 0, b.length, seed, output)

      case VariantUtil.Type.UUID =>
        val uuid = variant.getUuid()
        val s1 = XXH3.hashLong128FoldedInto(uuid.getMostSignificantBits, seed, output)
        XXH3.hashLong128FoldedInto(uuid.getLeastSignificantBits, mixPos(s1, 2), output)

      case VariantUtil.Type.OBJECT =>
        val n = variant.objectSize()
        var accum = 0L
        var i = 0
        while (i < n) {
          val field = variant.getFieldAtIndex(i)
          val kb = field.key.getBytes(java.nio.charset.StandardCharsets.UTF_8)
          val kh = XXH3.hashBytes64(kb, 0, kb.length, 0L)
          accum += hashVariantInto(field.value, kh, output)
          i += 1
        }
        XXH3.hashLong128FoldedInto(accum, seed ^ (n.toLong * POSITION_MIX), output)

      case VariantUtil.Type.ARRAY =>
        val n = variant.arraySize()
        var s = XXH3.hashInt128FoldedInto(n, seed, output)
        var pos = 1
        var i = 0
        while (i < n) {
          s = hashVariantInto(variant.getElementAtIndex(i), mixPos(s, pos), output)
          pos += 1
          i += 1
        }
        s
    }
  }
}

case class XxHash3(children: Seq[Expression], seed: Long) extends Expression
  with ImplicitCastInputTypes {

  def this(arguments: Seq[Expression]) = this(arguments, 0L)

  override def nullable: Boolean = false
  override def dataType: DataType = LongType
  override def inputTypes: Seq[AbstractDataType] = Seq.fill(children.length)(AnyDataType)
  override def foldable: Boolean = children.forall(_.foldable)
  override def contextIndependentFoldable: Boolean =
    children.forall(_.contextIndependentFoldable)
  override def prettyName: String = "xxh3_64"

  override def checkInputDataTypes(): TypeCheckResult = {
    if (children.length < 1) {
      throw QueryCompilationErrors.wrongNumArgsError(
        toSQLId(prettyName), Seq("> 0"), children.length)
    } else if (children.exists(child => !Xxh3HashCommon.supports(child.dataType))) {
      TypeCheckResult.TypeCheckFailure(
        s"$prettyName does not support input type " +
          children.find(child => !Xxh3HashCommon.supports(child.dataType)).get.dataType.sql)
    } else {
      TypeCheckResult.TypeCheckSuccess
    }
  }

  override def eval(input: InternalRow = null): Any = {
    var s = seed
    var pendingNulls = 0
    var nullRunStart = 0
    var pos = 1
    val len = children.length
    var i = 0
    while (i < len) {
      val v = children(i).eval(input)
      if (v != null) {
        s = Xxh3HashCommon.mixNullRun64(s, pendingNulls, nullRunStart)
        pendingNulls = 0
        s = Xxh3LongHashFunction.hash(v, children(i).dataType, Xxh3HashCommon.mixPos(s, pos),
          isCollationAware = false, legacyCollationAwareHashing = false)
      } else {
        if (pendingNulls == 0) {
          nullRunStart = pos
        }
        pendingNulls += 1
      }
      pos += 1
      i += 1
    }
    if (len > 1) {
      s = Xxh3HashCommon.mixNullRun64(s, pendingNulls, nullRunStart)
    }
    s
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val mixK = s"0x${Xxh3HashCommon.POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val hasher = classOf[XXH3].getName
    val seedLocal = ctx.freshName("hashSeed")
    val pendingNulls = ctx.addMutableState(
      CodeGenerator.JAVA_INT, "xxh3PendingNulls",
      v => s"$v = 0;", useFreshName = true)
    val nullRunStart = ctx.addMutableState(
      CodeGenerator.JAVA_INT, "xxh3NullRunStart",
      v => s"$v = 0;", useFreshName = true)

    def flushNulls: String =
      s"""|if ($pendingNulls > 0) {
          |  $seedLocal = $hasher.hashLong64((long) $pendingNulls,
          |    $seedLocal ^ (-((long) $nullRunStart) * $mixK));
          |  $pendingNulls = 0;
          |}""".stripMargin
    val flushTrailingNulls = if (children.length > 1) flushNulls else ""

    val perChild = children.zipWithIndex.map { case (child, idx) =>
      val pos = idx + 1
      val childGen = child.genCode(ctx)
      val mix = if (pos > 1) s"$seedLocal ^= ${pos}L * $mixK;" else ""
      val nonNullBody = XxHash3.genHash(
        ctx, childGen.value, child.dataType, seedLocal, legacyCollationAware = false)
      val body =
        s"""|$flushNulls
            |$mix
            |$nonNullBody""".stripMargin
      val step = if (child.nullable) {
        s"""|if (${childGen.isNull}) {
            |  if ($pendingNulls == 0) $nullRunStart = $pos;
            |  $pendingNulls++;
            |} else {
            |  $body
            |}""".stripMargin
      } else {
        body
      }
      childGen.code.toString + step
    }

    val splitChain = ctx.splitExpressionsWithCurrentInputs(
      expressions = perChild,
      funcName = "xxh3ChainStep",
      extraArguments = Seq("long" -> seedLocal),
      returnType = "long",
      makeSplitFunction = body =>
        s"""|$body
            |return $seedLocal;""".stripMargin,
      foldFunctions = _.map(call => s"$seedLocal = $call;").mkString("\n"))

    ev.copy(
      code = code"""
        |long $seedLocal = ${seed}L;
        |$pendingNulls = 0;
        |$nullRunStart = 0;
        |$splitChain
        |$flushTrailingNulls
        |long ${ev.value} = $seedLocal;
      """.stripMargin,
      isNull = FalseLiteral)
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): XxHash3 =
    copy(children = newChildren)
}

object XxHash3 {
  import Xxh3HashCommon.POSITION_MIX

  private[expressions] def genHash(
      ctx: CodegenContext,
      input: String,
      dt: DataType,
      seedVar: String,
      legacyCollationAware: Boolean): String = {
    val hasher = classOf[XXH3].getName
    def step(rhs: String): String = s"$seedVar = $rhs;"
    dt match {
      case NullType => ""

      case BooleanType => step(s"$hasher.hashInt64(($input) ? 1 : 0, $seedVar)")
      case ByteType | ShortType | IntegerType | DateType | _: YearMonthIntervalType =>
        step(s"$hasher.hashInt64($input, $seedVar)")
      case LongType | TimestampType | TimestampNTZType | _: DayTimeIntervalType | _: TimeType =>
        step(s"$hasher.hashLong64($input, $seedVar)")

      case _: TimestampNTZNanosType | _: TimestampLTZNanosType =>
        s"""|${step(s"$hasher.hashLong64($input.epochMicros, $seedVar)")}
            |${step(s"$hasher.hashInt64($input.nanosWithinMicro, $seedVar)")}
          """.stripMargin

      case FloatType =>
        s"""|if ($input == -0.0f) {
            |  ${step(s"$hasher.hashInt64(0, $seedVar)")}
            |} else {
            |  ${step(s"$hasher.hashInt64(Float.floatToIntBits($input), $seedVar)")}
            |}""".stripMargin

      case DoubleType =>
        s"""|if ($input == -0.0d) {
            |  ${step(s"$hasher.hashLong64(0L, $seedVar)")}
            |} else {
            |  ${step(s"$hasher.hashLong64(Double.doubleToLongBits($input), $seedVar)")}
            |}""".stripMargin

      case d: DecimalType =>
        if (d.precision <= Decimal.MAX_LONG_DIGITS) {
          step(s"$hasher.hashLong64($input.toUnscaledLong(), $seedVar)")
        } else {
          val bytes = ctx.freshName("decBytes")
          s"""|final byte[] $bytes = $input.toJavaBigDecimal().unscaledValue().toByteArray();
              |${step(s"$hasher.hashBytes64($bytes, 0, $bytes.length, $seedVar)")}
            """.stripMargin
        }

      case CalendarIntervalType =>
        s"""|${step(s"$hasher.hashLong64($input.microseconds, $seedVar)")}
            |${step(s"$hasher.hashInt64($input.days, $seedVar)")}
            |${step(s"$hasher.hashInt64($input.months, $seedVar)")}
          """.stripMargin

      case BinaryType =>
        step(s"$hasher.hashBytes64($input, 0, $input.length, $seedVar)")

      case _: GeometryType | _: GeographyType =>
        step(s"$hasher.hashUnsafeBytes64($input.getBaseObject(), " +
          s"$input.getBaseOffset(), $input.numBytes(), $seedVar)")

      case st: StringType =>
        genHashString(ctx, st, input, seedVar, legacyCollationAware)

      case ArrayType(et, containsNull) =>
        genHashForArray(ctx, input, et, containsNull, seedVar, legacyCollationAware)

      case st @ StructType(fields) if Xxh3HashCommon.useInterpretedStructHash(fields) =>
        val interpRef = ctx.addReferenceObj(
          "xxh3LongStructInterp", Xxh3LongHashFunction,
          Xxh3LongHashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3LongStructDt", st, classOf[DataType].getName)
        step(s"$interpRef.hash($input, $dtRef, $seedVar, false, $legacyCollationAware)")

      case StructType(fields) =>
        genHashForStruct(ctx, input, fields, seedVar, legacyCollationAware)

      case udt: UserDefinedType[_] =>
        genHash(ctx, input, udt.sqlType, seedVar, legacyCollationAware)

      case MapType(keyType, valueType, valueContainsNull) =>
        genHashForMap(
          ctx, input, keyType, valueType, valueContainsNull, seedVar, legacyCollationAware)

      case _: VariantType =>
        val interpRef = ctx.addReferenceObj(
          "xxh3LongInterp", Xxh3LongHashFunction, Xxh3LongHashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3LongDt", dt, classOf[DataType].getName)
        step(s"$interpRef.hash($input, $dtRef, $seedVar, false, $legacyCollationAware)")

      case _ =>
        throw new IllegalArgumentException(s"Unsupported XXH3 input type: $dt")
    }
  }

  private def genHashString(
      ctx: CodegenContext,
      stringType: StringType,
      input: String,
      seedVar: String,
      legacyCollationAware: Boolean): String = {
    val hasher = classOf[XXH3].getName
    val offset = "org.apache.spark.unsafe.Platform.BYTE_ARRAY_OFFSET"
    if (stringType.supportsBinaryEquality) {
      s"$seedVar = $hasher.hashUnsafeBytes64($input.getBaseObject(), " +
        s"$input.getBaseOffset(), $input.numBytes(), $seedVar);"
    } else if (legacyCollationAware) {
      val collation = CollationFactory.fetchCollation(stringType.collationId)
      val stringHash = ctx.freshName("stringHash")
      if (collation.isUtf8BinaryType || collation.isUtf8LcaseType) {
        s"""|long $stringHash = org.apache.spark.unsafe.types.UTF8String.fromBytes(
            |  (byte[]) org.apache.spark.sql.catalyst.util.CollationFactory
            |    .fetchCollation(${stringType.collationId}).sortKeyFunction.apply($input))
            |  .hashCode();
            |$seedVar = $hasher.hashLong64($stringHash, $seedVar);""".stripMargin
      } else if (collation.supportsSpaceTrimming) {
        s"""|long $stringHash = org.apache.spark.sql.catalyst.util.CollationFactory
            |  .fetchCollation(${stringType.collationId}).getCollator()
            |  .getCollationKey($input.trimRight().toValidString()).hashCode();
            |$seedVar = $hasher.hashLong64($stringHash, $seedVar);""".stripMargin
      } else {
        s"""|long $stringHash = org.apache.spark.sql.catalyst.util.CollationFactory
            |  .fetchCollation(${stringType.collationId}).getCollator()
            |  .getCollationKey($input.toValidString()).hashCode();
            |$seedVar = $hasher.hashLong64($stringHash, $seedVar);""".stripMargin
      }
    } else {
      s"$seedVar = $hasher.hashUnsafeBytes64($input.getBaseObject(), " +
        s"$input.getBaseOffset(), $input.numBytes(), $seedVar);"
    }
  }

  private def genHashForArray(
      ctx: CodegenContext,
      input: String,
      elementType: DataType,
      containsNull: Boolean,
      seedVar: String,
      legacyCollationAware: Boolean): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val n = ctx.freshName("n")
    val idx = ctx.freshName("idx")
    val pos = ctx.freshName("pos")
    val elem = ctx.freshName("elem")
    val jt = CodeGenerator.javaType(elementType)
    val elemHash = genHash(ctx, elem, elementType, seedVar, legacyCollationAware)
    val perElement = if (containsNull) {
      s"""|if (!$input.isNullAt($idx)) {
          |  if ($pos > 1) $seedVar ^= ((long) $pos) * $mixK;
          |  final $jt $elem = ${CodeGenerator.getValue(input, elementType, idx)};
          |  $elemHash
          |}""".stripMargin
    } else {
      s"""|if ($pos > 1) $seedVar ^= ((long) $pos) * $mixK;
          |final $jt $elem = ${CodeGenerator.getValue(input, elementType, idx)};
          |$elemHash""".stripMargin
    }
    s"""|{
        |  final int $n = $input.numElements();
        |  $seedVar = $hasher.hashInt64($n, $seedVar);
        |  int $pos = 1;
        |  for (int $idx = 0; $idx < $n; $idx++) {
        |    $perElement
        |    $pos++;
        |  }
        |}""".stripMargin
  }

  private def genHashForStruct(
      ctx: CodegenContext,
      input: String,
      fields: Array[StructField],
      seedVar: String,
      legacyCollationAware: Boolean): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val tmp = ctx.freshName("struct")
    val fieldsCode = fields.zipWithIndex.map { case (f, i) =>
      val pos = i + 1
      val v = ctx.freshName("field")
      val jt = CodeGenerator.javaType(f.dataType)
      val fieldHash = genHash(ctx, v, f.dataType, seedVar, legacyCollationAware)
      val mix = if (pos > 1) s"$seedVar ^= ${pos}L * $mixK;" else ""
      if (f.nullable) {
        s"""|if (!$tmp.isNullAt($i)) {
            |  $mix
            |  final $jt $v = ${CodeGenerator.getValue(tmp, f.dataType, i.toString)};
            |  $fieldHash
            |}""".stripMargin
      } else {
        s"""|{
            |  $mix
            |  final $jt $v = ${CodeGenerator.getValue(tmp, f.dataType, i.toString)};
            |  $fieldHash
            |}""".stripMargin
      }
    }.mkString("\n")
    s"""|{
        |  final org.apache.spark.sql.catalyst.InternalRow $tmp = $input;
        |  $seedVar = $hasher.hashInt64(${fields.length}, $seedVar);
        |  $fieldsCode
        |}""".stripMargin
  }

  private def genHashForMap(
      ctx: CodegenContext,
      input: String,
      keyType: DataType,
      valueType: DataType,
      valueContainsNull: Boolean,
      seedVar: String,
      legacyCollationAware: Boolean): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val n = ctx.freshName("mapSize")
    val keys = ctx.freshName("mapKeys")
    val values = ctx.freshName("mapValues")
    val idx = ctx.freshName("mapIndex")
    val key = ctx.freshName("mapKey")
    val value = ctx.freshName("mapValue")
    val entrySeed = ctx.freshName("mapEntrySeed")
    val accum = ctx.freshName("mapAccum")
    val keyHash = genHash(ctx, key, keyType, entrySeed, legacyCollationAware)
    val valueHash = genHash(ctx, value, valueType, entrySeed, legacyCollationAware)
    val valueCode = if (valueContainsNull) {
      s"""|if (!$values.isNullAt($idx)) {
          |  final ${CodeGenerator.javaType(valueType)} $value =
          |    ${CodeGenerator.getValue(values, valueType, idx)};
          |  $valueHash
          |}""".stripMargin
    } else {
      s"""|final ${CodeGenerator.javaType(valueType)} $value =
          |  ${CodeGenerator.getValue(values, valueType, idx)};
          |$valueHash""".stripMargin
    }
    s"""|{
        |  final int $n = $input.numElements();
        |  final org.apache.spark.sql.catalyst.util.ArrayData $keys = $input.keyArray();
        |  final org.apache.spark.sql.catalyst.util.ArrayData $values = $input.valueArray();
        |  long $accum = 0L;
        |  for (int $idx = 0; $idx < $n; $idx++) {
        |    long $entrySeed = 0L;
        |    final ${CodeGenerator.javaType(keyType)} $key =
        |      ${CodeGenerator.getValue(keys, keyType, idx)};
        |    $keyHash
        |    $valueCode
        |    $accum += $entrySeed;
        |  }
        |  $seedVar = $hasher.hashLong64($accum, $seedVar ^ ((long) $n * $mixK));
        |}""".stripMargin
  }
}

case class XxHash128(children: Seq[Expression], seed: Long) extends Expression
  with ImplicitCastInputTypes {

  def this(arguments: Seq[Expression]) = this(arguments, 0L)

  override def nullable: Boolean = false
  override def dataType: DataType = BinaryType
  override def inputTypes: Seq[AbstractDataType] = Seq.fill(children.length)(AnyDataType)
  override def foldable: Boolean = children.forall(_.foldable)
  override def contextIndependentFoldable: Boolean =
    children.forall(_.contextIndependentFoldable)
    override def prettyName: String = "xxh3_128"

  override def checkInputDataTypes(): TypeCheckResult = {
    if (children.length < 1) {
      throw QueryCompilationErrors.wrongNumArgsError(
        toSQLId(prettyName), Seq("> 0"), children.length)
    } else if (children.exists(child => !Xxh3HashCommon.supports(child.dataType))) {
      TypeCheckResult.TypeCheckFailure(
        s"$prettyName does not support input type " +
          children.find(child => !Xxh3HashCommon.supports(child.dataType)).get.dataType.sql)
    } else {
      TypeCheckResult.TypeCheckSuccess
    }
  }

  override def eval(input: InternalRow = null): Any = {
    var s = seed
    val result = new Array[Byte](16)
    var pendingNulls = 0
    var nullRunStart = 0
    var pos = 1
    val len = children.length
    var i = 0
    while (i < len) {
      val v = children(i).eval(input)
      if (v != null) {
        if (pendingNulls > 0) {
          s = Xxh3HashCommon.mixNullRun128Into(s, pendingNulls, nullRunStart, result)
          pendingNulls = 0
        }
        s = Xxh3Bytes128HashFunction.hashInto(
          v, children(i).dataType, Xxh3HashCommon.mixPos(s, pos), result)
      } else {
        if (pendingNulls == 0) {
          nullRunStart = pos
        }
        pendingNulls += 1
      }
      pos += 1
      i += 1
    }
    if (len > 1 && pendingNulls > 0) {
      Xxh3HashCommon.mixNullRun128Into(s, pendingNulls, nullRunStart, result)
    }
    result
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val mixK = s"0x${Xxh3HashCommon.POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val hasher = classOf[XXH3].getName
    val seedLocal = ctx.freshName("hashSeed")
    val resultLocal = ctx.freshName("xxh128Result")
    val pendingNulls = ctx.addMutableState(
      CodeGenerator.JAVA_INT, "xxh128PendingNulls",
      v => s"$v = 0;", useFreshName = true)
    val nullRunStart = ctx.addMutableState(
      CodeGenerator.JAVA_INT, "xxh128NullRunStart",
      v => s"$v = 0;", useFreshName = true)

    def flushNulls: String =
      s"""|if ($pendingNulls > 0) {
          |  $seedLocal = $hasher.hashLong128FoldedInto((long) $pendingNulls,
          |    $seedLocal ^ (-((long) $nullRunStart) * $mixK), $resultLocal);
          |  $pendingNulls = 0;
          |}""".stripMargin
    val flushTrailingNulls = if (children.length > 1) flushNulls else ""

    val perChild = children.zipWithIndex.map { case (child, idx) =>
      val pos = idx + 1
      val childGen = child.genCode(ctx)
      val mix = if (pos > 1) s"$seedLocal ^= ${pos}L * $mixK;" else ""
      val nonNullBody =
        XxHash128.genHash(ctx, childGen.value, child.dataType, resultLocal, seedLocal)
      val body =
        s"""|$flushNulls
            |$mix
            |$nonNullBody""".stripMargin
      val step = if (child.nullable) {
        s"""|if (${childGen.isNull}) {
            |  if ($pendingNulls == 0) $nullRunStart = $pos;
            |  $pendingNulls++;
            |} else {
            |  $body
            |}""".stripMargin
      } else {
        body
      }
      childGen.code.toString + step
    }

    val splitChain = ctx.splitExpressionsWithCurrentInputs(
      expressions = perChild,
      funcName = "xxh128ChainStep",
      extraArguments = Seq("long" -> seedLocal, "byte[]" -> resultLocal),
      returnType = "long",
      makeSplitFunction = body =>
        s"""|$body
            |return $seedLocal;""".stripMargin,
      foldFunctions = _.map(call => s"$seedLocal = $call;").mkString("\n"))

    ev.copy(
      code = code"""
        |long $seedLocal = ${seed}L;
        |final byte[] $resultLocal = new byte[16];
        |$pendingNulls = 0;
        |$nullRunStart = 0;
        |$splitChain
        |$flushTrailingNulls
        |byte[] ${ev.value} = $resultLocal;
      """.stripMargin,
      isNull = FalseLiteral)
  }

  override protected def withNewChildrenInternal(
      newChildren: IndexedSeq[Expression]): XxHash128 =
    copy(children = newChildren)
}

object XxHash128 {
  import Xxh3HashCommon.POSITION_MIX

  private[expressions] def genHash(
      ctx: CodegenContext,
      input: String,
      dt: DataType,
      resultVar: String,
      seedVar: String): String = {
    val hasher = classOf[XXH3].getName
    def step(method: String, args: String): String =
      s"$seedVar = $hasher.$method($args, $seedVar, $resultVar);"
    dt match {
      case NullType => ""

      case BooleanType => step("hashInt128FoldedInto", s"($input) ? 1 : 0")
      case ByteType | ShortType | IntegerType | DateType | _: YearMonthIntervalType =>
        step("hashInt128FoldedInto", input)
      case LongType | TimestampType | TimestampNTZType | _: DayTimeIntervalType | _: TimeType =>
        step("hashLong128FoldedInto", input)

      case _: TimestampNTZNanosType | _: TimestampLTZNanosType =>
        s"""|${step("hashLong128FoldedInto", s"$input.epochMicros")}
          |${step("hashInt128FoldedInto", s"$input.nanosWithinMicro")}
          """.stripMargin

      case FloatType =>
        s"""|if ($input == -0.0f) {
            |  ${step("hashInt128FoldedInto", "0")}
            |} else {
            |  ${step("hashInt128FoldedInto", s"Float.floatToIntBits($input)")}
            |}""".stripMargin

      case DoubleType =>
        s"""|if ($input == -0.0d) {
            |  ${step("hashLong128FoldedInto", "0L")}
            |} else {
            |  ${step("hashLong128FoldedInto", s"Double.doubleToLongBits($input)")}
            |}""".stripMargin

      case d: DecimalType =>
        if (d.precision <= Decimal.MAX_LONG_DIGITS) {
          step("hashLong128FoldedInto", s"$input.toUnscaledLong()")
        } else {
          val bytes = ctx.freshName("decBytes")
          s"""|final byte[] $bytes = $input.toJavaBigDecimal().unscaledValue().toByteArray();
              |${step("hashBytes128FoldedInto", s"$bytes, 0, $bytes.length")}
            """.stripMargin
        }

      case CalendarIntervalType =>
        s"""|${step("hashLong128FoldedInto", s"$input.microseconds")}
          |${step("hashInt128FoldedInto", s"$input.days")}
          |${step("hashInt128FoldedInto", s"$input.months")}
          """.stripMargin

      case BinaryType =>
        step("hashBytes128FoldedInto", s"$input, 0, $input.length")

      case _: GeometryType | _: GeographyType =>
        step("hashUnsafeBytes128FoldedInto",
          s"$input.getBaseObject(), $input.getBaseOffset(), $input.numBytes()")

      case _: StringType =>
        step("hashUnsafeBytes128FoldedInto",
          s"$input.getBaseObject(), $input.getBaseOffset(), $input.numBytes()")

      case ArrayType(et, containsNull) =>
        genHashForArray(ctx, input, et, containsNull, resultVar, seedVar)

      case st @ StructType(fields) if Xxh3HashCommon.useInterpretedStructHash(fields) =>
        val interpRef = ctx.addReferenceObj(
          "xxh3Bytes128StructInterp", Xxh3Bytes128HashFunction,
          Xxh3Bytes128HashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3Bytes128StructDt", st, classOf[DataType].getName)
        s"$seedVar = $interpRef.hashInto($input, $dtRef, $seedVar, $resultVar);"

      case StructType(fields) =>
        genHashForStruct(ctx, input, fields, resultVar, seedVar)

      case udt: UserDefinedType[_] =>
        genHash(ctx, input, udt.sqlType, resultVar, seedVar)

      case MapType(keyType, valueType, valueContainsNull) =>
        genHashForMap(
          ctx, input, keyType, valueType, valueContainsNull, resultVar, seedVar)

      case _: VariantType =>
        val interpRef = ctx.addReferenceObj(
          "xxh3Bytes128Interp", Xxh3Bytes128HashFunction,
          Xxh3Bytes128HashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3Bytes128Dt", dt, classOf[DataType].getName)
        s"$seedVar = $interpRef.hashInto($input, $dtRef, $seedVar, $resultVar);"

      case _ =>
        throw new IllegalArgumentException(s"Unsupported XXH3 input type: $dt")
    }
  }

  private def genHashForArray(
      ctx: CodegenContext,
      input: String,
      elementType: DataType,
      containsNull: Boolean,
      resultVar: String,
      seedVar: String): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val n = ctx.freshName("n")
    val idx = ctx.freshName("idx")
    val pos = ctx.freshName("pos")
    val elem = ctx.freshName("elem")
    val jt = CodeGenerator.javaType(elementType)
    val elemHash = genHash(ctx, elem, elementType, resultVar, seedVar)
    val perElement = if (containsNull) {
      s"""|if (!$input.isNullAt($idx)) {
          |  if ($pos > 1) $seedVar ^= ((long) $pos) * $mixK;
          |  final $jt $elem = ${CodeGenerator.getValue(input, elementType, idx)};
          |  $elemHash
          |}""".stripMargin
    } else {
      s"""|if ($pos > 1) $seedVar ^= ((long) $pos) * $mixK;
          |final $jt $elem = ${CodeGenerator.getValue(input, elementType, idx)};
          |$elemHash""".stripMargin
    }
    s"""|{
        |  final int $n = $input.numElements();
        |  $seedVar = $hasher.hashInt128FoldedInto($n, $seedVar, $resultVar);
        |  int $pos = 1;
        |  for (int $idx = 0; $idx < $n; $idx++) {
        |    $perElement
        |    $pos++;
        |  }
        |}""".stripMargin
  }

  private def genHashForStruct(
      ctx: CodegenContext,
      input: String,
      fields: Array[StructField],
      resultVar: String,
      seedVar: String): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val tmp = ctx.freshName("struct")
    val fieldsCode = fields.zipWithIndex.map { case (f, i) =>
      val pos = i + 1
      val v = ctx.freshName("field")
      val jt = CodeGenerator.javaType(f.dataType)
      val fieldHash = genHash(ctx, v, f.dataType, resultVar, seedVar)
      val mix = if (pos > 1) s"$seedVar ^= ${pos}L * $mixK;" else ""
      if (f.nullable) {
        s"""|if (!$tmp.isNullAt($i)) {
            |  $mix
            |  final $jt $v = ${CodeGenerator.getValue(tmp, f.dataType, i.toString)};
            |  $fieldHash
            |}""".stripMargin
      } else {
        s"""|{
            |  $mix
            |  final $jt $v = ${CodeGenerator.getValue(tmp, f.dataType, i.toString)};
            |  $fieldHash
            |}""".stripMargin
      }
    }.mkString("\n")
    s"""|{
        |  final org.apache.spark.sql.catalyst.InternalRow $tmp = $input;
        |  $seedVar = $hasher.hashInt128FoldedInto(
        |    ${fields.length}, $seedVar, $resultVar);
        |  $fieldsCode
        |}""".stripMargin
  }

  private def genHashForMap(
      ctx: CodegenContext,
      input: String,
      keyType: DataType,
      valueType: DataType,
      valueContainsNull: Boolean,
      resultVar: String,
      seedVar: String): String = {
    val hasher = classOf[XXH3].getName
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase(Locale.ROOT)}L"
    val n = ctx.freshName("mapSize")
    val keys = ctx.freshName("mapKeys")
    val values = ctx.freshName("mapValues")
    val idx = ctx.freshName("mapIndex")
    val key = ctx.freshName("mapKey")
    val value = ctx.freshName("mapValue")
    val entrySeed = ctx.freshName("mapEntrySeed")
    val accum = ctx.freshName("mapAccum")
    val keyHash = genHash(ctx, key, keyType, resultVar, entrySeed)
    val valueHash = genHash(ctx, value, valueType, resultVar, entrySeed)
    val valueCode = if (valueContainsNull) {
      s"""|if (!$values.isNullAt($idx)) {
          |  final ${CodeGenerator.javaType(valueType)} $value =
          |    ${CodeGenerator.getValue(values, valueType, idx)};
          |  $valueHash
          |}""".stripMargin
    } else {
      s"""|final ${CodeGenerator.javaType(valueType)} $value =
          |  ${CodeGenerator.getValue(values, valueType, idx)};
          |$valueHash""".stripMargin
    }
    s"""|{
        |  final int $n = $input.numElements();
        |  final org.apache.spark.sql.catalyst.util.ArrayData $keys = $input.keyArray();
        |  final org.apache.spark.sql.catalyst.util.ArrayData $values = $input.valueArray();
        |  long $accum = 0L;
        |  for (int $idx = 0; $idx < $n; $idx++) {
        |    long $entrySeed = 0L;
        |    final ${CodeGenerator.javaType(keyType)} $key =
        |      ${CodeGenerator.getValue(keys, keyType, idx)};
        |    $keyHash
        |    $valueCode
        |    $accum += $entrySeed;
        |  }
        |  $seedVar = $hasher.hashLong128FoldedInto(
        |    $accum, $seedVar ^ ((long) $n * $mixK), $resultVar);
        |}""".stripMargin
  }
}

@ExpressionDescription(
  usage = "_FUNC_(expr1, expr2, ...) - Returns a 64-bit xxHash3 hash of the arguments. " +
    "A null unary argument returns null. With multiple arguments, consecutive null values " +
    "are hashed as null runs, so null placement and input width perturb the result. " +
    "Hash seed defaults to 0.",
  examples = """
    Examples:
      > SELECT _FUNC_('Spark', array(123), 2);
       -1744444949335395721
  """,
  since = "4.3.0",
  group = "hash_funcs")
case class Xxh364 private(
    arguments: Seq[Expression],
    replacement: Expression)
  extends RuntimeReplaceable with InheritAnalysisRules {

  def this(arguments: Seq[Expression]) = this(arguments, With(arguments: _*) { refs =>
    if (refs.length == 1 && Xxh364.usesRawBytes(refs.head.dataType)) Xxh364Binary(refs.head)
    else if (refs.length == 1) Xxh364Structural(refs.head)
    else XxHash3(refs, 0L)
  })

  override def parameters: Seq[Expression] = arguments

  override def prettyName: String = "xxh3_64"

  override protected def withNewChildInternal(newChild: Expression): Xxh364 =
    copy(replacement = newChild)
}

object Xxh364 {
  private def usesRawBytes(dataType: DataType): Boolean = dataType match {
    case _: StringType | BinaryType | NullType => true
    case _ => false
  }

  def apply(arguments: Seq[Expression]): Xxh364 = new Xxh364(arguments)
}

@ExpressionDescription(
  usage = "_FUNC_(expr1, expr2, ...) - Returns a 128-bit xxHash3 hash of the arguments as a " +
    "16-byte binary value. A null unary argument returns null. With multiple arguments, " +
    "consecutive null values are hashed as null runs, so null placement and input width perturb " +
    "the result. Hash seed defaults to 0. String values are always hashed as raw UTF-8 bytes " +
    "regardless of collation.",
  examples = """
    Examples:
      > SELECT hex(_FUNC_('Spark'));
       7D57DD84C60C86CA1F4E82AB91A12B5E
  """,
  since = "4.3.0",
  group = "hash_funcs")
case class Xxh3128 private(
    arguments: Seq[Expression],
    replacement: Expression)
  extends RuntimeReplaceable with InheritAnalysisRules {

  def this(arguments: Seq[Expression]) = this(arguments, With(arguments: _*) { refs =>
    if (refs.length == 1 && Xxh3128.usesRawBytes(refs.head.dataType)) Xxh3128Binary(refs.head)
    else if (refs.length == 1) Xxh3128Structural(refs.head)
    else XxHash128(refs, 0L)
  })

  override def parameters: Seq[Expression] = arguments

  override def prettyName: String = "xxh3_128"

  override protected def withNewChildInternal(newChild: Expression): Xxh3128 =
    copy(replacement = newChild)
}

object Xxh3128 {
  private def usesRawBytes(dataType: DataType): Boolean = dataType match {
    case _: StringType | BinaryType | NullType => true
    case _ => false
  }

  def apply(arguments: Seq[Expression]): Xxh3128 = new Xxh3128(arguments)
}

@ExpressionDescription(
  usage = "_FUNC_(expr1, expr2, ...) - Returns a 128-bit xxHash3 hash of the arguments as a " +
    "32-character lowercase hexadecimal string.",
  examples = """
    Examples:
      > SELECT _FUNC_('Spark');
       7d57dd84c60c86ca1f4e82ab91a12b5e
  """,
  since = "4.3.0",
  group = "hash_funcs")
case class Xxh3128Hex private(
    arguments: Seq[Expression],
    replacement: Expression)
  extends RuntimeReplaceable with InheritAnalysisRules {

  def this(arguments: Seq[Expression]) =
    this(arguments,
      if (arguments.isEmpty) Xxh3128HexEmpty else Lower(Hex(Xxh3128(arguments))))

  override def parameters: Seq[Expression] = arguments

  override def prettyName: String = "xxh3_128_hex"

  override protected def withNewChildInternal(newChild: Expression): Xxh3128Hex =
    copy(replacement = newChild)
}

object Xxh3128Hex {
  def apply(arguments: Seq[Expression]): Xxh3128Hex = new Xxh3128Hex(arguments)
}

private case object Xxh3128HexEmpty extends LeafExpression with CodegenFallback {
  override def nullable: Boolean = false
  override def dataType: DataType = StringType
  override def checkInputDataTypes(): TypeCheckResult =
    throw QueryCompilationErrors.wrongNumArgsError(
      toSQLId("xxh3_128_hex"), Seq("> 0"), 0)
  override def eval(input: InternalRow): Any =
    throw new IllegalStateException("xxh3_128_hex arity must be checked before evaluation")
}

private case class Xxh364Structural(child: Expression) extends UnaryExpression {
  override def nullIntolerant: Boolean = true
  override def dataType: DataType = LongType
  override protected def nullSafeEval(input: Any): Any =
    Xxh3LongHashFunction.hash(input, child.dataType, 0L)
  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode =
    nullSafeCodeGen(ctx, ev, value =>
      s"""|${ev.value} = 0L;
          |${XxHash3.genHash(
            ctx, value, child.dataType, ev.value, legacyCollationAware = false)}
        """.stripMargin)
  override protected def withNewChildInternal(newChild: Expression): Xxh364Structural =
    copy(child = newChild)
}

private case class Xxh3128Structural(child: Expression) extends UnaryExpression {
  override def nullIntolerant: Boolean = true
  override def dataType: DataType = BinaryType
  override protected def nullSafeEval(input: Any): Any =
    Xxh3Bytes128HashFunction.hash(input, child.dataType, 0L)
  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    nullSafeCodeGen(ctx, ev, value => {
      val seed = ctx.freshName("hashSeed")
      s"""|long $seed = 0L;
          |${ev.value} = new byte[16];
          |${XxHash128.genHash(ctx, value, child.dataType, ev.value, seed)}
        """.stripMargin
    })
  }
  override protected def withNewChildInternal(newChild: Expression): Xxh3128Structural =
    copy(child = newChild)
}

private case class Xxh364Binary(child: Expression)
  extends UnaryExpression with ImplicitCastInputTypes {
  override def nullIntolerant: Boolean = true
  override def dataType: DataType = LongType
  override def inputTypes: Seq[DataType] = Seq(BinaryType)
  override protected def nullSafeEval(input: Any): Any =
    XXH3.hash64(input.asInstanceOf[Array[Byte]])
  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val cls = classOf[XXH3].getName
    nullSafeCodeGen(ctx, ev, value => s"${ev.value} = $cls.hash64($value);")
  }
  override protected def withNewChildInternal(newChild: Expression): Xxh364Binary =
    copy(child = newChild)
}

private case class Xxh3128Binary(child: Expression)
  extends UnaryExpression with ImplicitCastInputTypes {
  override def nullIntolerant: Boolean = true
  override def dataType: DataType = BinaryType
  override def inputTypes: Seq[DataType] = Seq(BinaryType)
  override protected def nullSafeEval(input: Any): Any = {
    val bytes = input.asInstanceOf[Array[Byte]]
    XXH3.hashBytes128(bytes, 0, bytes.length, 0L)
  }
  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val cls = classOf[XXH3].getName
    nullSafeCodeGen(ctx, ev, value =>
      s"${ev.value} = $cls.hashBytes128($value, 0, $value.length, 0L);")
  }
  override protected def withNewChildInternal(newChild: Expression): Xxh3128Binary =
    copy(child = newChild)
}
