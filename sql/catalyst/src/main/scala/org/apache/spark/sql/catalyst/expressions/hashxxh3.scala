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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.Cast._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.util.{ArrayData, CollationFactory, MapData}
import org.apache.spark.sql.errors.QueryCompilationErrors
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.types.variant.{Variant, VariantUtil}
import org.apache.spark.unsafe.types.{CalendarInterval, TimestampNanosVal, UTF8String, VariantVal}

////////////////////////////////////////////////////////////////////////////////////////////////////
// XXH3 family hash expressions: xxhash3 (64-bit) and xxhash128 (128-bit).
//
// Diverges from the xxhash64 / Murmur3Hash family in three deliberate ways:
//
//   1. Position-aware nulls. The chain advances a column-position counter for every child
//      (including nulls). Non-null children past position 1 mix the position into the seed via
//      seed ^= pos * 0x9E3779B97F4A7C15L (golden-ratio prime). The next primitive hash call's
//      avalanche fully distributes the perturbation. Cost: ~1 ns per non-null child past the
//      first. Properties:
//        - (x, null, y) != (x, y)            (interior null perturbs)
//        - (null, x)    != (x)               (leading null perturbs)
//        - (x, y, null) == (x, y)            (trailing null invariant; counter advances, no emit)
//        - xxhash3(x)   == XXH3.hashXxx64(x, 0)  (single-value interop with the C reference)
//
//   2. MapType and VariantType are accepted. Maps and Variant objects use O(n) addition-based
//      accumulation: each entry is hashed independently (at seed 0) and the results summed, so
//      insertion order does not affect the result. {x: 0, y: 1} == {y: 1, x: 0}.
//
//   3. All-null inputs return a tombstone (0L for xxhash3, 16 zero bytes for xxhash128) which no
//      real input can reach in practice.
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Shared utilities for the XXH3-family interpreted dispatch.
 *
 * Position mixing uses the golden-ratio prime; XOR-mixing into the chain seed is fully avalanched
 * by the subsequent primitive hash call, so the cost is one multiply + one XOR per non-null child
 * past position 1.
 */
private[expressions] object Xxh3HashCommon {
  final val POSITION_MIX: Long = 0x9E3779B97F4A7C15L

  /** Returns seed perturbed by position p (1-indexed); identity at p == 1. */
  @inline def mixPos(seed: Long, p: Int): Long =
    if (p > 1) seed ^ (p.toLong * POSITION_MIX) else seed

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64-bit dispatch
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Interpreted dispatch for xxhash3 (64-bit). Returns the next chain seed directly as a Long.
 *
 * The hash() method must be called with a non-null value. Caller is responsible for chain
 * management (position-mix, null skipping).
 */
object Xxh3LongHashFunction extends Serializable {
  import Xxh3HashCommon._

  /**
   * Hash a single non-null value. Returns the new chain seed.
   *
   * @param value     the value to hash (must be non-null)
   * @param dataType  the Spark SQL type of `value`
   * @param seed      the chain seed flowing in (already position-mixed by caller for top-level)
   * @param isCollationAware  see [[InterpretedHashFunction]]; only meaningful for [[StringType]]
   * @param legacyCollationAwareHashing  see [[InterpretedHashFunction]]
   */
  def hash(
      value: Any,
      dataType: DataType,
      seed: Long,
      isCollationAware: Boolean = false,
      legacyCollationAwareHashing: Boolean = false): Long = {
    value match {
      case b: Boolean => XXH3.hashInt64(if (b) 1 else 0, seed)
      case b: Byte    => XXH3.hashInt64(b, seed)
      case s: Short   => XXH3.hashInt64(s, seed)
      case i: Int     => XXH3.hashInt64(i, seed)
      case l: Long    => XXH3.hashLong64(l, seed)

      case f: Float if f == -0.0f => XXH3.hashInt64(0, seed)
      case f: Float => XXH3.hashInt64(java.lang.Float.floatToIntBits(f), seed)

      case d: Double if d == -0.0d => XXH3.hashLong64(0L, seed)
      case d: Double => XXH3.hashLong64(java.lang.Double.doubleToLongBits(d), seed)

      case d: Decimal =>
        val precision = dataType.asInstanceOf[DecimalType].precision
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

      case s: UTF8String =>
        hashString(s, dataType.asInstanceOf[StringType], seed,
          isCollationAware, legacyCollationAwareHashing)

      case array: ArrayData =>
        val elementType = dataType match {
          case udt: UserDefinedType[_] => udt.sqlType.asInstanceOf[ArrayType].elementType
          case ArrayType(et, _) => et
        }
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
        val (kt, vt) = dataType match {
          case udt: UserDefinedType[_] =>
            val mt = udt.sqlType.asInstanceOf[MapType]
            mt.keyType -> mt.valueType
          case MapType(kt, vt, _) => kt -> vt
        }
        val n = map.numElements()
        val keys = map.keyArray()
        val values = map.valueArray()
        // O(n) addition-based accumulation: hash each (key, value) pair at a fixed seed (0L)
        // so pairs are independent of each other and can be summed in any order.
        // Binding value to key via hash(v, seed=hash(k, 0L)) ensures {k1:v2,k2:v1} hashes
        // differently from {k1:v1,k2:v2}. Addition never cancels duplicate entries to zero
        // (unlike XOR), so malformed maps with repeated keys remain distinguishable.
        var accum = 0L
        var i = 0
        while (i < n) {
          val k = keys.get(i, kt)
          val kh = hash(k, kt, 0L, isCollationAware, legacyCollationAwareHashing)
          val v = if (values.isNullAt(i)) null else values.get(i, vt)
          accum += (if (v != null)
            hash(v, vt, kh, isCollationAware, legacyCollationAwareHashing)
          else kh)
          i += 1
        }
        XXH3.hashLong64(accum, seed ^ (n.toLong * POSITION_MIX))

      case row: InternalRow =>
        val types: Array[DataType] = dataType match {
          case udt: UserDefinedType[_] =>
            udt.sqlType.asInstanceOf[StructType].map(_.dataType).toArray
          case StructType(fields) => fields.map(_.dataType)
        }
        val len = row.numFields
        // Length-prefix so struct<a> != struct<a,b>(_, null) at the same outer position.
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
    }
  }

  /**
   * Hash a UTF8String, honouring the existing xxhash64 collation rules. For binary-equality
   * collations (the default UTF8_BINARY), raw bytes are hashed. For non-binary collations,
   * behaviour mirrors [[InterpretedHashFunction]].
   */
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

  /**
   * Recursively hash a Variant value by its logical type. The internal binary format of Variant
   * is explicitly transient, so we never hash its raw bytes; OBJECT fields use addition-based
   * accumulation, ARRAYs are length-prefixed and position-chained, primitives dispatch by type.
   */
  private def hashVariant(variant: Variant, seed: Long): Long = {
    import Xxh3HashCommon._
    variant.getType() match {
      case VariantUtil.Type.NULL    => XXH3.hashLong64(0L, seed)
      case VariantUtil.Type.BOOLEAN => XXH3.hashInt64(if (variant.getBoolean()) 1 else 0, seed)
      case VariantUtil.Type.LONG    => XXH3.hashLong64(variant.getLong(), seed)
      case VariantUtil.Type.DATE    => XXH3.hashInt64(variant.getLong().toInt, seed)
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
        // O(n) addition-based accumulation: Variant object fields are unordered and
        // the spec guarantees no duplicate keys. Addition is consistent with MapType
        // and ensures field-value swaps are detected: {a:1,b:2} != {a:2,b:1}.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// 128-bit dispatch
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Interpreted dispatch for xxhash128 (128-bit). Returns the 16-byte hash of the value.
 *
 * The chain folds 16-byte results back to a Long seed via [[XXH3.fold128]] between steps;
 * caller is responsible for chain management (position-mix, null skipping). Returns the last
 * 16-byte primitive emission; if a container produced multiple primitive calls internally, the
 * last call's bytes are returned (the chain seed already encodes the full history).
 */
object Xxh3Bytes128HashFunction extends Serializable {
  import Xxh3HashCommon._

  /** Hash a single non-null value. Returns 16 bytes. */
  def hash(value: Any, dataType: DataType, seed: Long): Array[Byte] = value match {
    case b: Boolean => XXH3.hashInt128(if (b) 1 else 0, seed)
    case b: Byte    => XXH3.hashInt128(b, seed)
    case s: Short   => XXH3.hashInt128(s, seed)
    case i: Int     => XXH3.hashInt128(i, seed)
    case l: Long    => XXH3.hashLong128(l, seed)

    case f: Float if f == -0.0f => XXH3.hashInt128(0, seed)
    case f: Float => XXH3.hashInt128(java.lang.Float.floatToIntBits(f), seed)

    case d: Double if d == -0.0d => XXH3.hashLong128(0L, seed)
    case d: Double => XXH3.hashLong128(java.lang.Double.doubleToLongBits(d), seed)

    case d: Decimal =>
      val precision = dataType.asInstanceOf[DecimalType].precision
      if (precision <= Decimal.MAX_LONG_DIGITS) {
        XXH3.hashLong128(d.toUnscaledLong, seed)
      } else {
        val bytes = d.toJavaBigDecimal.unscaledValue().toByteArray
        XXH3.hashBytes128(bytes, 0, bytes.length, seed)
      }

    case c: CalendarInterval =>
      val h1 = XXH3.hashLong128(c.microseconds, seed)
      val h2 = XXH3.hashInt128(c.days, XXH3.fold128(h1))
      XXH3.hashInt128(c.months, XXH3.fold128(h2))

    case t: TimestampNanosVal =>
      val h1 = XXH3.hashLong128(t.epochMicros, seed)
      XXH3.hashInt128(t.nanosWithinMicro, XXH3.fold128(h1))

    case a: Array[Byte] =>
      XXH3.hashBytes128(a, 0, a.length, seed)

    case s: UTF8String =>
      XXH3.hashUnsafeBytes128(s.getBaseObject, s.getBaseOffset, s.numBytes, seed)

    case array: ArrayData =>
      val elementType = dataType match {
        case udt: UserDefinedType[_] => udt.sqlType.asInstanceOf[ArrayType].elementType
        case ArrayType(et, _) => et
      }
      val n = array.numElements()
      var h = XXH3.hashInt128(n, seed)
      var pos = 1
      var i = 0
      while (i < n) {
        if (!array.isNullAt(i)) {
          h = hash(array.get(i, elementType), elementType, mixPos(XXH3.fold128(h), pos))
        }
        pos += 1
        i += 1
      }
      h

    case map: MapData =>
      val (kt, vt) = dataType match {
        case udt: UserDefinedType[_] =>
          val mt = udt.sqlType.asInstanceOf[MapType]
          mt.keyType -> mt.valueType
        case MapType(kt, vt, _) => kt -> vt
      }
      val n = map.numElements()
      val keys = map.keyArray()
      val values = map.valueArray()
      // Same addition-based pair accumulation as the 64-bit function.
      // We fold each 128-bit pair hash to a Long for accumulation, then finalize
      // with hashLong128 so the output is a full 128-bit digest.
      var accum = 0L
      var i = 0
      while (i < n) {
        val k = keys.get(i, kt)
        val kh = XXH3.fold128(hash(k, kt, 0L))
        val v = if (values.isNullAt(i)) null else values.get(i, vt)
        accum += (if (v != null) XXH3.fold128(hash(v, vt, kh)) else kh)
        i += 1
      }
      XXH3.hashLong128(accum, seed ^ (n.toLong * POSITION_MIX))

    case row: InternalRow =>
      val types: Array[DataType] = dataType match {
        case udt: UserDefinedType[_] =>
          udt.sqlType.asInstanceOf[StructType].map(_.dataType).toArray
        case StructType(fields) => fields.map(_.dataType)
      }
      val len = row.numFields
      var h = XXH3.hashInt128(len, seed)
      var pos = 1
      var i = 0
      while (i < len) {
        if (!row.isNullAt(i)) {
          h = hash(row.get(i, types(i)), types(i), mixPos(XXH3.fold128(h), pos))
        }
        pos += 1
        i += 1
      }
      h

    case v: VariantVal =>
      hashVariant(new Variant(v.getValue(), v.getMetadata()), seed)
  }

  private def hashVariant(variant: Variant, seed: Long): Array[Byte] = {
    variant.getType() match {
      case VariantUtil.Type.NULL    => XXH3.hashLong128(0L, seed)
      case VariantUtil.Type.BOOLEAN => XXH3.hashInt128(if (variant.getBoolean()) 1 else 0, seed)
      case VariantUtil.Type.LONG    => XXH3.hashLong128(variant.getLong(), seed)
      case VariantUtil.Type.DATE    => XXH3.hashInt128(variant.getLong().toInt, seed)
      case VariantUtil.Type.TIMESTAMP | VariantUtil.Type.TIMESTAMP_NTZ =>
        XXH3.hashLong128(variant.getLong(), seed)

      case VariantUtil.Type.DOUBLE =>
        val d = variant.getDouble()
        if (d == -0.0d) XXH3.hashLong128(0L, seed)
        else XXH3.hashLong128(java.lang.Double.doubleToLongBits(d), seed)

      case VariantUtil.Type.FLOAT =>
        val f = variant.getFloat()
        if (f == -0.0f) XXH3.hashInt128(0, seed)
        else XXH3.hashInt128(java.lang.Float.floatToIntBits(f), seed)

      case VariantUtil.Type.DECIMAL =>
        val bytes = variant.getDecimal().unscaledValue().toByteArray
        XXH3.hashBytes128(bytes, 0, bytes.length, seed)

      case VariantUtil.Type.STRING =>
        val s = UTF8String.fromString(variant.getString())
        XXH3.hashUnsafeBytes128(s.getBaseObject, s.getBaseOffset, s.numBytes, seed)

      case VariantUtil.Type.BINARY =>
        val b = variant.getBinary()
        XXH3.hashBytes128(b, 0, b.length, seed)

      case VariantUtil.Type.UUID =>
        val uuid = variant.getUuid()
        val h1 = XXH3.hashLong128(uuid.getMostSignificantBits, seed)
        XXH3.hashLong128(uuid.getLeastSignificantBits, mixPos(XXH3.fold128(h1), 2))

      case VariantUtil.Type.OBJECT =>
        // O(n) addition-based accumulation, consistent with the 64-bit variant and map handling.
        val n = variant.objectSize()
        var accum = 0L
        var i = 0
        while (i < n) {
          val field = variant.getFieldAtIndex(i)
          val kb = field.key.getBytes(java.nio.charset.StandardCharsets.UTF_8)
          val kh = XXH3.hashBytes64(kb, 0, kb.length, 0L)
          accum += XXH3.fold128(hashVariant(field.value, kh))
          i += 1
        }
        XXH3.hashLong128(accum, seed ^ (n.toLong * POSITION_MIX))

      case VariantUtil.Type.ARRAY =>
        val n = variant.arraySize()
        var h = XXH3.hashInt128(n, seed)
        var pos = 1
        var i = 0
        while (i < n) {
          h = hashVariant(variant.getElementAtIndex(i), mixPos(XXH3.fold128(h), pos))
          pos += 1
          i += 1
        }
        h
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// xxhash3 (64-bit) expression
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * A xxHash3 64-bit hash expression. See file-level header for the family contract.
 */
@ExpressionDescription(
  usage = "_FUNC_(expr1, expr2, ...) - Returns a 64-bit xxHash3 hash of the arguments. " +
    "Null values advance position but contribute no primitive hash; interior nulls perturb " +
    "the result, trailing nulls are invariant. All-null inputs return 0. Hash seed defaults to 0.",
  examples = """
    Examples:
      > SELECT _FUNC_('Spark', array(123), 2);
       -1744444949335395721
  """,
  since = "4.3.0",
  group = "hash_funcs")
case class XxHash3(children: Seq[Expression], seed: Long) extends Expression
  with ImplicitCastInputTypes {

  def this(arguments: Seq[Expression]) = this(arguments, 0L)

  override def nullable: Boolean = false
  override def dataType: DataType = LongType
  override def inputTypes: Seq[AbstractDataType] = Seq.fill(children.length)(AnyDataType)
  override def foldable: Boolean = children.forall(_.foldable)
  override def contextIndependentFoldable: Boolean =
    children.forall(_.contextIndependentFoldable)
  override def prettyName: String = "xxhash3"

  override def checkInputDataTypes(): TypeCheckResult = {
    if (children.length < 1) {
      throw QueryCompilationErrors.wrongNumArgsError(
        toSQLId(prettyName), Seq("> 0"), children.length)
    } else {
      TypeCheckResult.TypeCheckSuccess
    }
  }

  private lazy val legacyCollationAwareHashing: Boolean =
    SQLConf.get.getConf(SQLConf.COLLATION_AWARE_HASHING_ENABLED)

  override def eval(input: InternalRow = null): Any = {
    var s = seed
    var emitted = false
    var pos = 1
    val len = children.length
    var i = 0
    while (i < len) {
      val v = children(i).eval(input)
      if (v != null) {
        s = Xxh3LongHashFunction.hash(v, children(i).dataType, Xxh3HashCommon.mixPos(s, pos),
          isCollationAware = false, legacyCollationAwareHashing = legacyCollationAwareHashing)
        emitted = true
      }
      pos += 1
      i += 1
    }
    if (!emitted) seed else s
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val mixK = s"0x${Xxh3HashCommon.POSITION_MIX.toHexString.toUpperCase}L"
    val seedLocal = ctx.freshName("hashSeed")
    // `emitted` is the "any child contributed?" flag; it cannot be threaded through
    // splitExpressions (single accumulator only), so we materialise it as a per-row
    // mutable field and reset it at the start of each eval.
    val emittedField = ctx.addMutableState(
      CodeGenerator.JAVA_BOOLEAN, "xxh3Emitted",
      v => s"$v = false;", useFreshName = true)

    val perChild = children.zipWithIndex.map { case (child, idx) =>
      val pos = idx + 1
      val childGen = child.genCode(ctx)
      val mix = if (pos > 1) s"$seedLocal ^= ${pos}L * $mixK;" else ""
      val nonNullBody =
        XxHash3.genHash(ctx, childGen.value, child.dataType, seedLocal, legacyCollationAwareHashing)
      val body =
        s"""|$mix
            |$nonNullBody
            |$emittedField = true;""".stripMargin
      childGen.code.toString + ctx.nullSafeExec(child.nullable, childGen.isNull)(body)
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
        |$emittedField = false;
        |$splitChain
        |long ${ev.value} = $emittedField ? $seedLocal : ${seed}L;
      """.stripMargin,
      isNull = FalseLiteral)
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): XxHash3 =
    copy(children = newChildren)
}

object XxHash3 {
  import Xxh3HashCommon.POSITION_MIX

  /** Emit a single primitive-or-container hash step that updates `seedVar` for the given type. */
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

      case st: StringType =>
        genHashString(ctx, st, input, seedVar, legacyCollationAware)

      case ArrayType(et, containsNull) =>
        genHashForArray(ctx, input, et, containsNull, seedVar, legacyCollationAware)

      case StructType(fields) =>
        genHashForStruct(ctx, input, fields, seedVar, legacyCollationAware)

      case udt: UserDefinedType[_] =>
        genHash(ctx, input, udt.sqlType, seedVar, legacyCollationAware)

      case _: MapType | _: VariantType =>
        // Delegate to interpreted dispatch: addition-based accumulation and Variant recursion
        // are too complex to inline cleanly into generated Java.
        val interpRef = ctx.addReferenceObj(
          "xxh3LongInterp", Xxh3LongHashFunction, Xxh3LongHashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3LongDt", dt, classOf[DataType].getName)
        step(s"$interpRef.hash($input, $dtRef, $seedVar, false, $legacyCollationAware)")
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
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase}L"
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
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase}L"
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// xxhash128 (128-bit) expression
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * A xxHash3 128-bit hash expression returning a 16-byte big-endian [high64 || low64] digest.
 * See file-level header for the family contract.
 */
@ExpressionDescription(
  usage = "_FUNC_(expr1, expr2, ...) - Returns a 128-bit xxHash3 hash of the arguments as a " +
    "16-byte binary value. Null values advance position but contribute no primitive hash; " +
    "interior nulls perturb the result, trailing nulls are invariant. All-null inputs return " +
    "16 zero bytes. Hash seed defaults to 0. String values are always hashed as raw UTF-8 " +
    "bytes regardless of collation.",
  examples = """
    Examples:
      > SELECT hex(_FUNC_('Spark'));
       7D57DD84C60C86CA1F4E82AB91A12B5E
  """,
  since = "4.3.0",
  group = "hash_funcs")
case class XxHash128(children: Seq[Expression], seed: Long) extends Expression
  with ImplicitCastInputTypes {

  def this(arguments: Seq[Expression]) = this(arguments, 0L)

  override def nullable: Boolean = false
  override def dataType: DataType = BinaryType
  override def inputTypes: Seq[AbstractDataType] = Seq.fill(children.length)(AnyDataType)
  override def foldable: Boolean = children.forall(_.foldable)
  override def contextIndependentFoldable: Boolean =
    children.forall(_.contextIndependentFoldable)
  override def prettyName: String = "xxhash128"

  override def checkInputDataTypes(): TypeCheckResult = {
    if (children.length < 1) {
      throw QueryCompilationErrors.wrongNumArgsError(
        toSQLId(prettyName), Seq("> 0"), children.length)
    } else {
      TypeCheckResult.TypeCheckSuccess
    }
  }

  override def eval(input: InternalRow = null): Any = {
    var s = seed
    var result: Array[Byte] = null
    var pos = 1
    val len = children.length
    var i = 0
    while (i < len) {
      val v = children(i).eval(input)
      if (v != null) {
        val r = Xxh3Bytes128HashFunction.hash(v, children(i).dataType,
          Xxh3HashCommon.mixPos(s, pos))
        result = r
        s = XXH3.fold128(r)
      }
      pos += 1
      i += 1
    }
    if (result == null) new Array[Byte](16) else result
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val mixK = s"0x${Xxh3HashCommon.POSITION_MIX.toHexString.toUpperCase}L"
    val seedLocal = ctx.freshName("hashSeed")
    // The 16-byte result is the OUT accumulator; it cannot be threaded through
    // splitExpressions (single accumulator), so we materialise it as a per-row
    // mutable field reset at the start of each eval.
    val resultField = ctx.addMutableState(
      "byte[]", "xxh128Result",
      v => s"$v = null;", useFreshName = true)

    val perChild = children.zipWithIndex.map { case (child, idx) =>
      val pos = idx + 1
      val childGen = child.genCode(ctx)
      val mix = if (pos > 1) s"$seedLocal ^= ${pos}L * $mixK;" else ""
      val nonNullBody =
        XxHash128.genHash(ctx, childGen.value, child.dataType, resultField, seedLocal)
      val body =
        s"""|$mix
            |$nonNullBody""".stripMargin
      childGen.code.toString + ctx.nullSafeExec(child.nullable, childGen.isNull)(body)
    }

    val splitChain = ctx.splitExpressionsWithCurrentInputs(
      expressions = perChild,
      funcName = "xxh128ChainStep",
      extraArguments = Seq("long" -> seedLocal),
      returnType = "long",
      makeSplitFunction = body =>
        s"""|$body
            |return $seedLocal;""".stripMargin,
      foldFunctions = _.map(call => s"$seedLocal = $call;").mkString("\n"))

    ev.copy(
      code = code"""
        |long $seedLocal = ${seed}L;
        |$resultField = null;
        |$splitChain
        |byte[] ${ev.value} = ($resultField == null) ? new byte[16] : $resultField;
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
    def step(rhs: String): String =
      s"""|$resultVar = $rhs;
          |$seedVar = $hasher.fold128($resultVar);""".stripMargin
    dt match {
      case NullType => ""

      case BooleanType => step(s"$hasher.hashInt128(($input) ? 1 : 0, $seedVar)")
      case ByteType | ShortType | IntegerType | DateType | _: YearMonthIntervalType =>
        step(s"$hasher.hashInt128($input, $seedVar)")
      case LongType | TimestampType | TimestampNTZType | _: DayTimeIntervalType | _: TimeType =>
        step(s"$hasher.hashLong128($input, $seedVar)")

      case _: TimestampNTZNanosType | _: TimestampLTZNanosType =>
        s"""|${step(s"$hasher.hashLong128($input.epochMicros, $seedVar)")}
            |${step(s"$hasher.hashInt128($input.nanosWithinMicro, $seedVar)")}
          """.stripMargin

      case FloatType =>
        s"""|if ($input == -0.0f) {
            |  ${step(s"$hasher.hashInt128(0, $seedVar)")}
            |} else {
            |  ${step(s"$hasher.hashInt128(Float.floatToIntBits($input), $seedVar)")}
            |}""".stripMargin

      case DoubleType =>
        s"""|if ($input == -0.0d) {
            |  ${step(s"$hasher.hashLong128(0L, $seedVar)")}
            |} else {
            |  ${step(s"$hasher.hashLong128(Double.doubleToLongBits($input), $seedVar)")}
            |}""".stripMargin

      case d: DecimalType =>
        if (d.precision <= Decimal.MAX_LONG_DIGITS) {
          step(s"$hasher.hashLong128($input.toUnscaledLong(), $seedVar)")
        } else {
          val bytes = ctx.freshName("decBytes")
          s"""|final byte[] $bytes = $input.toJavaBigDecimal().unscaledValue().toByteArray();
              |${step(s"$hasher.hashBytes128($bytes, 0, $bytes.length, $seedVar)")}
            """.stripMargin
        }

      case CalendarIntervalType =>
        s"""|${step(s"$hasher.hashLong128($input.microseconds, $seedVar)")}
            |${step(s"$hasher.hashInt128($input.days, $seedVar)")}
            |${step(s"$hasher.hashInt128($input.months, $seedVar)")}
          """.stripMargin

      case BinaryType =>
        step(s"$hasher.hashBytes128($input, 0, $input.length, $seedVar)")

      case _: StringType =>
        step(s"$hasher.hashUnsafeBytes128($input.getBaseObject(), $input.getBaseOffset(), " +
          s"$input.numBytes(), $seedVar)")

      case ArrayType(et, containsNull) =>
        genHashForArray(ctx, input, et, containsNull, resultVar, seedVar)

      case StructType(fields) =>
        genHashForStruct(ctx, input, fields, resultVar, seedVar)

      case udt: UserDefinedType[_] =>
        genHash(ctx, input, udt.sqlType, resultVar, seedVar)

      case _: MapType | _: VariantType =>
        val interpRef = ctx.addReferenceObj(
          "xxh3Bytes128Interp", Xxh3Bytes128HashFunction,
          Xxh3Bytes128HashFunction.getClass.getName)
        val dtRef = ctx.addReferenceObj("xxh3Bytes128Dt", dt, classOf[DataType].getName)
        step(s"$interpRef.hash($input, $dtRef, $seedVar)")
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
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase}L"
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
        |  $resultVar = $hasher.hashInt128($n, $seedVar);
        |  $seedVar = $hasher.fold128($resultVar);
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
    val mixK = s"0x${POSITION_MIX.toHexString.toUpperCase}L"
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
        |  $resultVar = $hasher.hashInt128(${fields.length}, $seedVar);
        |  $seedVar = $hasher.fold128($resultVar);
        |  $fieldsCode
        |}""".stripMargin
  }
}
