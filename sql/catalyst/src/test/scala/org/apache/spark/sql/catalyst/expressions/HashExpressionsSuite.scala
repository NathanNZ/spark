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

import java.nio.charset.StandardCharsets
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.{Duration, LocalTime, Period, ZoneId, ZoneOffset}

import scala.collection.mutable.ArrayBuffer
import scala.language.implicitConversions

import org.scalatest.exceptions.TestFailedException

import org.apache.spark.SparkFunSuite
import org.apache.spark.sql.{RandomDataGenerator, Row}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.{ExamplePointUDT, ExpressionEncoder}
import org.apache.spark.sql.catalyst.expressions.codegen.GenerateMutableProjection
import org.apache.spark.sql.catalyst.expressions.variant.VariantExpressionEvalUtils
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, CollationFactory, DateTimeUtils, GenericArrayData, IntervalUtils}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ArrayType, StructType, _}
import org.apache.spark.unsafe.hash.Murmur3_x86_32
import org.apache.spark.unsafe.types.{TimestampNanosVal, UTF8String, VariantVal}
import org.apache.spark.util.ArrayImplicits._

class HashExpressionsSuite extends SparkFunSuite with ExpressionEvalHelper {
  val random = new scala.util.Random
  implicit def stringToUTF8Str(str: String): UTF8String = UTF8String.fromString(str)

  test("md5") {
    checkEvaluation(Md5(Literal("ABC".getBytes(StandardCharsets.UTF_8))),
      "902fbdd2b1df0c4f70b4a5d23525e932")
    checkEvaluation(Md5(Literal.create(Array[Byte](1, 2, 3, 4, 5, 6), BinaryType)),
      "6ac1e56bc78f031059be7be854522c4c")
    checkEvaluation(Md5(Literal.create(null, BinaryType)), null)
    checkConsistencyBetweenInterpretedAndCodegen(Md5, BinaryType)
  }

  test("sha1") {
    checkEvaluation(Sha1(Literal("ABC".getBytes(StandardCharsets.UTF_8))),
      "3c01bdbb26f358bab27f267924aa2c9a03fcfdb8")
    checkEvaluation(Sha1(Literal.create(Array[Byte](1, 2, 3, 4, 5, 6), BinaryType)),
      "5d211bad8f4ee70e16c7d343a838fc344a1ed961")
    checkEvaluation(Sha1(Literal.create(null, BinaryType)), null)
    checkEvaluation(Sha1(Literal("".getBytes(StandardCharsets.UTF_8))),
      "da39a3ee5e6b4b0d3255bfef95601890afd80709")
    checkConsistencyBetweenInterpretedAndCodegen(Sha1, BinaryType)
  }

  test("sha2") {
    checkEvaluation(Sha2(Literal("ABC".getBytes(StandardCharsets.UTF_8)), Literal(224)),
      "107c5072b799c4771f328304cfe1ebb375eb6ea7f35a3aa753836fad")
    checkEvaluation(Sha2(Literal("ABC".getBytes(StandardCharsets.UTF_8)), Literal(0)),
      "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78")
    checkEvaluation(Sha2(Literal("ABC".getBytes(StandardCharsets.UTF_8)), Literal(256)),
      "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78")
    checkEvaluation(Sha2(Literal.create(Array[Byte](1, 2, 3, 4, 5, 6), BinaryType), Literal(384)),
      "557cfe660c753b830efa61528fc350ef384a7a4b9d3467c6230049bc59548eb8" +
        "a404874baff89cb0f9bd18400829fdc2")
    checkEvaluation(Sha2(Literal("ABC".getBytes(StandardCharsets.UTF_8)), Literal(512)),
      "397118fdac8d83ad98813c50759c85b8c47565d8268bf10da483153b747a7474" +
        "3a58a90e85aa9f705ce6984ffc128db567489817e4092d050d8a1cc596ddc119")
    // unsupported bit length
    checkEvaluation(Sha2(Literal.create(null, BinaryType), Literal(1024)), null)
    // null input and valid bit length
    checkEvaluation(Sha2(Literal.create(null, BinaryType), Literal(512)), null)
    // valid input and null bit length
    checkEvaluation(Sha2(Literal("ABC".getBytes(StandardCharsets.UTF_8)),
      Literal.create(null, IntegerType)), null)
    checkEvaluation(Sha2(Literal.create(null, BinaryType), Literal.create(null, IntegerType)), null)
  }

  test("crc32") {
    checkEvaluation(Crc32(Literal("ABC".getBytes(StandardCharsets.UTF_8))), 2743272264L)
    checkEvaluation(Crc32(Literal.create(Array[Byte](1, 2, 3, 4, 5, 6), BinaryType)),
      2180413220L)
    checkEvaluation(Crc32(Literal.create(null, BinaryType)), null)
    checkConsistencyBetweenInterpretedAndCodegen(Crc32, BinaryType)
  }

  test("xxhash128 codegen parity for multi-child chains") {
    // Pairs of literals: each chain is 2-child to test that the seed correctly
    // threads from child N's fold into child N+1.
    def check(label: String, children: Seq[Literal]): Unit = {
      val expr = new XxHash128(children)
      val expected = expr.eval(null).asInstanceOf[Array[Byte]]
      try checkEvaluation(expr, expected) catch {
        case t: Throwable =>
          fail(s"multi-child chain '$label' failed: ${t.getMessage}")
      }
    }
    val udt = new ExamplePointUDT
    val point = udt.serialize(
      new org.apache.spark.sql.catalyst.encoders.ExamplePoint(1.5, 2.5))
    check("null+int", Seq(Literal.create(null, NullType), Literal.create(7, IntegerType)))
    check("int+null", Seq(Literal.create(7, IntegerType), Literal.create(null, NullType)))
    check("int+udt", Seq(Literal.create(7, IntegerType), Literal.create(point, udt)))
    check("udt+int", Seq(Literal.create(point, udt), Literal.create(7, IntegerType)))
    check("string+udt", Seq(
      Literal.create(UTF8String.fromString("hi"), StringType),
      Literal.create(point, udt)))
    check("null+null", Seq(Literal.create(null, NullType), Literal.create(null, NullType)))
  }

  test("xxhash128 codegen parity for UDT") {
    val udt = new ExamplePointUDT
    val point = udt.serialize(
      new org.apache.spark.sql.catalyst.encoders.ExamplePoint(1.5, 2.5))
    val expr = new XxHash128(Seq(Literal.create(point, udt)))
    val expected = expr.eval(null).asInstanceOf[Array[Byte]]
    checkEvaluation(expr, expected)
  }

  test("xxhash128 codegen parity per primitive type") {
    // Drive each primitive type independently so a mismatch fingerprints to one type.
    val cases: Seq[(String, Any, DataType)] = Seq(
      ("null", null, NullType),
      ("boolean", true, BooleanType),
      ("byte", (-12).toByte, ByteType),
      ("short", 32767.toShort, ShortType),
      ("int", 1390689586, IntegerType),
      ("long", 8663709074L, LongType),
      ("float", 1.5f, FloatType),
      ("float -0", -0.0f, FloatType),
      ("double", 6.0967941679987284, DoubleType),
      ("double -0", -0.0d, DoubleType),
      ("smallDec", Decimal(123456789L, 10, 0), DecimalType(10, 0)),
      ("bigDec", Decimal(BigDecimal("123456789.123456789"), 38, 18), DecimalType(38, 18)),
      ("string", UTF8String.fromString("hello"), StringType),
      ("binary", Array[Byte](1, 2, 3, 4, 5), BinaryType),
      ("date", 19000, DateType),
      ("timestamp", 1700000000000000L, TimestampType))
    for ((name, value, dt) <- cases) {
      val expr = new XxHash128(Seq(Literal.create(value, dt)))
      val expected = expr.eval(null).asInstanceOf[Array[Byte]]
      try checkEvaluation(expr, expected) catch {
        case t: Throwable =>
          fail(s"xxhash128 parity failed for case '$name' (dt=$dt, value=$value): ${t.getMessage}")
      }
    }
  }

  test("xxhash128") {
    def expectedBytes(high64: Long, low64: Long): Array[Byte] = {
      ByteBuffer.allocate(16).order(ByteOrder.BIG_ENDIAN).putLong(high64).putLong(low64).array()
    }

    checkEvaluation(
      new XxHash128(Seq(Literal.create(Array.empty[Byte], BinaryType))),
      expectedBytes(-7374073936536430376L, 6918025063187695999L))
    checkEvaluation(
      new XxHash128(Seq(Literal.create(Array[Byte](0), BinaryType))),
      expectedBytes(-6427377105285148822L, -4302098779834749733L))
  }

  test("xxhash3") {
    checkEvaluation(new XxHash3(Seq(Literal.create(Array.empty[Byte], BinaryType))),
      3244421341483603138L)
    checkEvaluation(new XxHash3(Seq(Literal.create(Array[Byte](0), BinaryType))),
      -4302098779834749733L)
  }

  // ---------------------------------------------------------------------------
  // XXH3-family contract: position-aware nulls, single-value interop, tombstone,
  // Map insertion-order independence, Variant field-order independence.
  // ---------------------------------------------------------------------------

  private def xxh3Eval(children: Seq[Expression]): Long =
    new XxHash3(children).eval(InternalRow.empty).asInstanceOf[Long]

  private def xxh128Eval(children: Seq[Expression]): Array[Byte] =
    new XxHash128(children).eval(InternalRow.empty).asInstanceOf[Array[Byte]]

  test("xxhash3 single-value interop with the C reference") {
    // hash(x) at top-level position 1 has no position-mix and must match the bare XXH3 call.
    val intLit = Literal.create(1234, IntegerType)
    assert(xxh3Eval(Seq(intLit)) == XXH3.hashInt64(1234, 0L))
  }

  test("xxhash128 single-value interop with the C reference") {
    val byteArr = Literal.create(Array[Byte](1, 2, 3, 4, 5), BinaryType)
    assert(java.util.Arrays.equals(
      xxh128Eval(Seq(byteArr)),
      XXH3.hashUnsafeBytes128(Array[Byte](1, 2, 3, 4, 5), 16L, 5, 0L)))
  }

  test("xxhash3 interior null perturbs the result") {
    val x = Literal.create(1, IntegerType)
    val y = Literal.create(2, IntegerType)
    val z = Literal.create(null, IntegerType)
    assert(xxh3Eval(Seq(x, z, y)) != xxh3Eval(Seq(x, y)))
    assert(xxh3Eval(Seq(z, x))    != xxh3Eval(Seq(x)))
    assert(xxh3Eval(Seq(z, z, x)) != xxh3Eval(Seq(z, x)))
  }

  test("xxhash128 interior null perturbs the result") {
    val x = Literal.create(1, IntegerType)
    val y = Literal.create(2, IntegerType)
    val z = Literal.create(null, IntegerType)
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(x, z, y)), xxh128Eval(Seq(x, y))))
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(z, x)),    xxh128Eval(Seq(x))))
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(z, z, x)), xxh128Eval(Seq(z, x))))
  }

  test("xxhash3 trailing nulls are invariant") {
    val x = Literal.create(1, IntegerType)
    val y = Literal.create(2, IntegerType)
    val z = Literal.create(null, IntegerType)
    assert(xxh3Eval(Seq(x, y)) == xxh3Eval(Seq(x, y, z)))
    assert(xxh3Eval(Seq(x, y)) == xxh3Eval(Seq(x, y, z, z)))
  }

  test("xxhash128 trailing nulls are invariant") {
    val x = Literal.create(1, IntegerType)
    val y = Literal.create(2, IntegerType)
    val z = Literal.create(null, IntegerType)
    assert(java.util.Arrays.equals(xxh128Eval(Seq(x, y)), xxh128Eval(Seq(x, y, z))))
    assert(java.util.Arrays.equals(xxh128Eval(Seq(x, y)), xxh128Eval(Seq(x, y, z, z))))
  }

  test("xxhash3 null is distinct from 0L / false / 0.0 / 0.0f") {
    val z = Literal.create(null, LongType)
    val zNullInt = Literal.create(null, IntegerType)
    val l0 = Literal.create(0L, LongType)
    val i0 = Literal.create(0, IntegerType)
    val b0 = Literal.create(false, BooleanType)
    val d0 = Literal.create(0.0d, DoubleType)
    val f0 = Literal.create(0.0f, FloatType)
    // (null) is the all-null tombstone (returns seed=0 for xxhash3, 16 zero bytes for xxhash128).
    // Each non-null zero must hash to something non-zero (i.e. != tombstone).
    assert(xxh3Eval(Seq(l0))        != 0L)
    assert(xxh3Eval(Seq(i0))        != 0L)
    assert(xxh3Eval(Seq(b0))        != 0L)
    assert(xxh3Eval(Seq(d0))        != 0L)
    assert(xxh3Eval(Seq(f0))        != 0L)
    // Both null types (Long-null and Int-null) collapse to tombstone identically.
    assert(xxh3Eval(Seq(z))         == 0L)
    assert(xxh3Eval(Seq(zNullInt))  == 0L)
    // Mixed: (null, x) differs from (0L, x) -- null contributes no primitive call but advances pos.
    val x = Literal.create(7L, LongType)
    assert(xxh3Eval(Seq(z, x)) != xxh3Eval(Seq(l0, x)))
  }

  test("xxhash128 null is distinct from 0L") {
    val z  = Literal.create(null, LongType)
    val l0 = Literal.create(0L, LongType)
    val x  = Literal.create(7L, LongType)
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(z, x)), xxh128Eval(Seq(l0, x))))
  }

  test("xxhash3 all-null inputs return the tombstone") {
    val z = Literal.create(null, IntegerType)
    assert(xxh3Eval(Seq(z))       == 0L)
    assert(xxh3Eval(Seq(z, z, z)) == 0L)
  }

  test("xxhash128 all-null inputs return the tombstone") {
    val z = Literal.create(null, IntegerType)
    assert(java.util.Arrays.equals(xxh128Eval(Seq(z)),       new Array[Byte](16)))
    assert(java.util.Arrays.equals(xxh128Eval(Seq(z, z, z)), new Array[Byte](16)))
  }

  test("xxhash3 seed parity: explicit seed vs default seed") {
    val x = Literal.create(42L, LongType)
    assert(xxh3Eval(Seq(x)) == XxHash3(Seq(x), 0L).eval(InternalRow.empty))
    val seeded = XxHash3(Seq(x), 7L).eval(InternalRow.empty).asInstanceOf[Long]
    assert(seeded != xxh3Eval(Seq(x)))
  }

  test("xxhash128 seed parity: explicit seed vs default seed") {
    val x = Literal.create(42L, LongType)
    val seeded128 = XxHash128(Seq(x), 7L).eval(InternalRow.empty).asInstanceOf[Array[Byte]]
    assert(!java.util.Arrays.equals(seeded128, xxh128Eval(Seq(x))))
  }

  test("xxhash3 MapType: entries are insertion-order independent") {
    val k1 = UTF8String.fromString("alpha")
    val k2 = UTF8String.fromString("bravo")
    val k3 = UTF8String.fromString("charlie")
    val keysAsc  = new GenericArrayData(Array[Any](k1, k2, k3))
    val keysDesc = new GenericArrayData(Array[Any](k3, k2, k1))
    val valsAsc  = new GenericArrayData(Array[Any](1L, 2L, 3L))
    val valsDesc = new GenericArrayData(Array[Any](3L, 2L, 1L))
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    val mapAsc  = Literal.create(new ArrayBasedMapData(keysAsc, valsAsc), mt)
    val mapDesc = Literal.create(new ArrayBasedMapData(keysDesc, valsDesc), mt)
    assert(xxh3Eval(Seq(mapAsc)) == xxh3Eval(Seq(mapDesc)))
    val valsAltered = new GenericArrayData(Array[Any](1L, 2L, 99L))
    val mapAltered = Literal.create(new ArrayBasedMapData(keysAsc, valsAltered), mt)
    assert(xxh3Eval(Seq(mapAsc)) != xxh3Eval(Seq(mapAltered)))
  }

  test("xxhash128 MapType: entries are insertion-order independent") {
    val k1 = UTF8String.fromString("alpha")
    val k2 = UTF8String.fromString("bravo")
    val k3 = UTF8String.fromString("charlie")
    val keysAsc  = new GenericArrayData(Array[Any](k1, k2, k3))
    val keysDesc = new GenericArrayData(Array[Any](k3, k2, k1))
    val valsAsc  = new GenericArrayData(Array[Any](1L, 2L, 3L))
    val valsDesc = new GenericArrayData(Array[Any](3L, 2L, 1L))
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    val mapAsc  = Literal.create(new ArrayBasedMapData(keysAsc, valsAsc), mt)
    val mapDesc = Literal.create(new ArrayBasedMapData(keysDesc, valsDesc), mt)
    assert(java.util.Arrays.equals(xxh128Eval(Seq(mapAsc)), xxh128Eval(Seq(mapDesc))))
  }

  test("xxhash3 MapType: duplicate-key entries hash distinctly (addition, not XOR)") {
    // With XOR accumulation, entryHash(k,v) ^ entryHash(k,v) == 0 for any identical duplicate,
    // so {a:1,a:1} and {b:2,b:2} would both collapse to accumulator=0 -> same hash.
    // Addition does not have this flaw. This test guards against reverting to XOR.
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    def dupMap(k: String, v: Long) = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any](UTF8String.fromString(k), UTF8String.fromString(k))),
        new GenericArrayData(Array[Any](v, v))),
      mt)
    assert(xxh3Eval(Seq(dupMap("a", 1L))) != xxh3Eval(Seq(dupMap("b", 2L))),
      "{a:1,a:1} and {b:2,b:2} must not collide - XOR would make both reduce to 0")
    assert(xxh3Eval(Seq(dupMap("a", 1L))) != xxh3Eval(Seq(dupMap("a", 2L))),
      "{a:1,a:1} and {a:2,a:2} must differ")
  }

  test("xxhash128 MapType: duplicate-key entries hash distinctly (addition, not XOR)") {
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    def dupMap(k: String, v: Long) = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any](UTF8String.fromString(k), UTF8String.fromString(k))),
        new GenericArrayData(Array[Any](v, v))),
      mt)
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(dupMap("a", 1L))), xxh128Eval(Seq(dupMap("b", 2L)))),
      "{a:1,a:1} and {b:2,b:2} must not collide - XOR would make both reduce to 0")
  }

  test("xxhash3 MapType: multi-entry duplicates do not cancel (addition vs XOR)") {
    // XOR: each pair cancels, {a:1,b:2,a:1,b:2} reduces to 0 == {}.
    // Addition: 2*h(a,1)+2*h(b,2) != 0, and != h({a:1,b:2}).
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    val ka = UTF8String.fromString("a"); val kb = UTF8String.fromString("b")
    val fullDupMap = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any](ka, kb, ka, kb)),
        new GenericArrayData(Array[Any](1L, 2L, 1L, 2L))), mt)
    val singleMap = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any](ka, kb)),
        new GenericArrayData(Array[Any](1L, 2L))), mt)
    val emptyMap = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any]()),
        new GenericArrayData(Array[Any]())), mt)
    assert(xxh3Eval(Seq(fullDupMap)) != xxh3Eval(Seq(emptyMap)))
    assert(xxh3Eval(Seq(fullDupMap)) != xxh3Eval(Seq(singleMap)))
  }

  test("xxhash128 MapType: multi-entry duplicates do not cancel (addition vs XOR)") {
    val mt = MapType(StringType, LongType, valueContainsNull = false)
    val ka = UTF8String.fromString("a"); val kb = UTF8String.fromString("b")
    val fullDupMap = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any](ka, kb, ka, kb)),
        new GenericArrayData(Array[Any](1L, 2L, 1L, 2L))), mt)
    val emptyMap = Literal.create(
      new ArrayBasedMapData(
        new GenericArrayData(Array[Any]()),
        new GenericArrayData(Array[Any]())), mt)
    assert(!java.util.Arrays.equals(xxh128Eval(Seq(fullDupMap)), xxh128Eval(Seq(emptyMap))))
  }

  test("xxhash3 VariantType: object fields are name-order independent") {
    def parseVariant(json: String): VariantVal =
      VariantExpressionEvalUtils.parseJson(UTF8String.fromString(json))
    val abLit = Literal.create(parseVariant("""{"a":1,"b":2}"""), VariantType)
    val baLit = Literal.create(parseVariant("""{"b":2,"a":1}"""), VariantType)
    assert(xxh3Eval(Seq(abLit)) == xxh3Eval(Seq(baLit)))
    val altLit = Literal.create(parseVariant("""{"a":1,"b":99}"""), VariantType)
    assert(xxh3Eval(Seq(abLit)) != xxh3Eval(Seq(altLit)))
    val nestedAB = Literal.create(parseVariant("""{"x":{"a":1,"b":2},"y":3}"""), VariantType)
    val nestedBA = Literal.create(parseVariant("""{"y":3,"x":{"b":2,"a":1}}"""), VariantType)
    assert(xxh3Eval(Seq(nestedAB)) == xxh3Eval(Seq(nestedBA)))
  }

  test("xxhash128 VariantType: object fields are name-order independent") {
    def parseVariant(json: String): VariantVal =
      VariantExpressionEvalUtils.parseJson(UTF8String.fromString(json))
    val abLit = Literal.create(parseVariant("""{"a":1,"b":2}"""), VariantType)
    val baLit = Literal.create(parseVariant("""{"b":2,"a":1}"""), VariantType)
    assert(java.util.Arrays.equals(xxh128Eval(Seq(abLit)), xxh128Eval(Seq(baLit))))
    val nestedAB = Literal.create(parseVariant("""{"x":{"a":1,"b":2},"y":3}"""), VariantType)
    val nestedBA = Literal.create(parseVariant("""{"y":3,"x":{"b":2,"a":1}}"""), VariantType)
    assert(java.util.Arrays.equals(xxh128Eval(Seq(nestedAB)), xxh128Eval(Seq(nestedBA))))
  }

  test("xxhash3 codegen path matches interpreted for null patterns") {
    val x = Literal.create(7, IntegerType)
    val y = Literal.create(42L, LongType)
    val z = Literal.create(null, IntegerType)
    Seq(Seq(z, x, y), Seq(x, z, y), Seq(x, y, z), Seq(z, z, x), Seq(z, z, z, x), Seq(x, z, z, y))
      .foreach { children =>
        val expr = new XxHash3(children)
        checkEvaluation(expr, expr.eval(InternalRow.empty))
      }
  }

  test("xxhash128 codegen path matches interpreted for null patterns") {
    val x = Literal.create(7, IntegerType)
    val y = Literal.create(42L, LongType)
    val z = Literal.create(null, IntegerType)
    Seq(Seq(z, x, y), Seq(x, z, y), Seq(x, y, z), Seq(z, z, x), Seq(z, z, z, x), Seq(x, z, z, y))
      .foreach { children =>
        val expr = new XxHash128(children)
        checkEvaluation(expr, expr.eval(InternalRow.empty))
      }
  }

  def checkHiveHash(input: Any, dataType: DataType, expected: Long): Unit = {
    // Note : All expected hashes need to be computed using Hive 1.2.1
    val actual = HiveHashFunction.hash(
      input,
      dataType,
      seed = 0,
      isCollationAware = true,
      // legacyCollationAwareHashing only matters when isCollationAware is false.
      legacyCollationAwareHashing = false
    )

    withClue(s"hash mismatch for input = `$input` of type `$dataType`.") {
      assert(actual == expected)
    }
  }

  def checkHiveHashForIntegralType(dataType: DataType): Unit = {
    // corner cases
    checkHiveHash(null, dataType, 0)
    checkHiveHash(1, dataType, 1)
    checkHiveHash(0, dataType, 0)
    checkHiveHash(-1, dataType, -1)
    checkHiveHash(Int.MaxValue, dataType, Int.MaxValue)
    checkHiveHash(Int.MinValue, dataType, Int.MinValue)

    // random values
    for (_ <- 0 until 10) {
      val input = random.nextInt()
      checkHiveHash(input, dataType, input)
    }
  }

  test("hive-hash for null") {
    checkHiveHash(null, NullType, 0)
  }

  test("hive-hash for boolean") {
    checkHiveHash(true, BooleanType, 1)
    checkHiveHash(false, BooleanType, 0)
  }

  test("hive-hash for byte") {
    checkHiveHashForIntegralType(ByteType)
  }

  test("hive-hash for short") {
    checkHiveHashForIntegralType(ShortType)
  }

  test("hive-hash for int") {
    checkHiveHashForIntegralType(IntegerType)
  }

  test("hive-hash for long") {
    checkHiveHash(1L, LongType, 1L)
    checkHiveHash(0L, LongType, 0L)
    checkHiveHash(-1L, LongType, 0L)
    checkHiveHash(Long.MaxValue, LongType, -2147483648)
    // Hive's fails to parse this.. but the hashing function itself can handle this input
    checkHiveHash(Long.MinValue, LongType, -2147483648)

    for (_ <- 0 until 10) {
      val input = random.nextLong()
      checkHiveHash(input, LongType, ((input >>> 32) ^ input).toInt)
    }
  }

  test("hive-hash for float") {
    checkHiveHash(0F, FloatType, 0)
    checkHiveHash(0.0F, FloatType, 0)
    checkHiveHash(1.1F, FloatType, 1066192077L)
    checkHiveHash(-1.1F, FloatType, -1081291571)
    checkHiveHash(99999999.99999999999F, FloatType, 1287568416L)
    checkHiveHash(Float.MaxValue, FloatType, 2139095039)
    checkHiveHash(Float.MinValue, FloatType, -8388609)
  }

  test("hive-hash for double") {
    checkHiveHash(0, DoubleType, 0)
    checkHiveHash(0.0, DoubleType, 0)
    checkHiveHash(1.1, DoubleType, -1503133693)
    checkHiveHash(-1.1, DoubleType, 644349955)
    checkHiveHash(1000000000.000001, DoubleType, 1104006509)
    checkHiveHash(1000000000.0000000000000000000000001, DoubleType, 1104006501)
    checkHiveHash(9999999999999999999.9999999999999999999, DoubleType, 594568676)
    checkHiveHash(Double.MaxValue, DoubleType, -2146435072)
    checkHiveHash(Double.MinValue, DoubleType, 1048576)
  }

  test("hive-hash for string") {
    checkHiveHash(UTF8String.fromString("apache spark"), StringType, 1142704523L)
    checkHiveHash(UTF8String.fromString("!@#$%^&*()_+=-"), StringType, -613724358L)
    checkHiveHash(UTF8String.fromString("abcdefghijklmnopqrstuvwxyz"), StringType, 958031277L)
    checkHiveHash(UTF8String.fromString("AbCdEfGhIjKlMnOpQrStUvWxYz012"), StringType, -648013852L)
    // scalastyle:off nonascii
    checkHiveHash(UTF8String.fromString("数据砖头"), StringType, -898686242L)
    checkHiveHash(UTF8String.fromString("नमस्ते"), StringType, 2006045948L)
    // scalastyle:on nonascii
  }

  test("hive-hash for date type") {
    def checkHiveHashForDateType(dateString: String, expected: Long): Unit = {
      checkHiveHash(
        DateTimeUtils.stringToDate(UTF8String.fromString(dateString)).get,
        DateType,
        expected)
    }

    // basic case
    checkHiveHashForDateType("2017-01-01", 17167)

    // boundary cases
    checkHiveHashForDateType("0000-01-01", -719528)
    checkHiveHashForDateType("9999-12-31", 2932896)

    // epoch
    checkHiveHashForDateType("1970-01-01", 0)

    // before epoch
    checkHiveHashForDateType("1800-01-01", -62091)

    // negative year
    checkHiveHashForDateType("-1212-01-01", -1162202)

    // Invalid input: bad date string. Hive returns 0 for such cases
    intercept[NoSuchElementException](checkHiveHashForDateType("0-0-0", 0))
    intercept[NoSuchElementException](checkHiveHashForDateType("2016-99-99", 0))

    // Invalid input: Empty string. Hive returns 0 for this case
    intercept[NoSuchElementException](checkHiveHashForDateType("", 0))

    // Invalid input: February 30th for a leap year. Hive supports this but Spark doesn't
    intercept[NoSuchElementException](checkHiveHashForDateType("2016-02-30", 16861))
  }

  test("hive-hash for timestamp type") {
    def checkHiveHashForTimestampType(
        timestamp: String,
        expected: Long,
        zoneId: ZoneId = ZoneOffset.UTC): Unit = {
      checkHiveHash(
        DateTimeUtils.stringToTimestamp(UTF8String.fromString(timestamp), zoneId).get,
        TimestampType,
        expected)
    }

    // basic case
    checkHiveHashForTimestampType("2017-02-24 10:56:29", 1445725271)

    // with higher precision
    checkHiveHashForTimestampType("2017-02-24 10:56:29.111111", 1353936655)

    // with different timezone
    checkHiveHashForTimestampType("2017-02-24 10:56:29", 1445732471,
      DateTimeUtils.getZoneId("US/Pacific"))

    // boundary cases
    checkHiveHashForTimestampType("0001-01-01 00:00:00", 1645969984)
    checkHiveHashForTimestampType("9999-01-01 00:00:00", -1081818240)

    // epoch
    checkHiveHashForTimestampType("1970-01-01 00:00:00", 0)

    // before epoch
    checkHiveHashForTimestampType("1800-01-01 03:12:45", -267420885)

    // Invalid input: bad timestamp string. Hive returns 0 for such cases
    intercept[NoSuchElementException](checkHiveHashForTimestampType("0-0-0 0:0:0", 0))
    intercept[NoSuchElementException](checkHiveHashForTimestampType("-99-99-99 99:99:45", 0))
    intercept[NoSuchElementException](checkHiveHashForTimestampType("555555-55555-5555", 0))

    // Invalid input: Empty string. Hive returns 0 for this case
    intercept[NoSuchElementException](checkHiveHashForTimestampType("", 0))

    // Invalid input: February 30th is a leap year. Hive supports this but Spark doesn't
    intercept[NoSuchElementException](checkHiveHashForTimestampType("2016-02-30 00:00:00", 0))

    // Invalid input: Hive accepts upto 9 decimal place precision but Spark uses upto 6
    intercept[TestFailedException](checkHiveHashForTimestampType("2017-02-24 10:56:29.11111111", 0))
  }

  test("hive-hash for CalendarInterval type") {
    def checkHiveHashForIntervalType(interval: String, expected: Long): Unit = {
      checkHiveHash(IntervalUtils.stringToInterval(UTF8String.fromString(interval)),
        CalendarIntervalType, expected)
    }

    // ----- MICROSEC -----

    // basic case
    checkHiveHashForIntervalType("interval 1 microsecond", 24273)

    // negative
    checkHiveHashForIntervalType("interval -1 microsecond", 22273)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 microsecond", 23273)
    checkHiveHashForIntervalType("interval 999 microsecond", 1022273)
    checkHiveHashForIntervalType("interval -999 microsecond", -975727)

    // ----- MILLISEC -----

    // basic case
    checkHiveHashForIntervalType("interval 1 millisecond", 1023273)

    // negative
    checkHiveHashForIntervalType("interval -1 millisecond", -976727)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 millisecond", 23273)
    checkHiveHashForIntervalType("interval 999 millisecond", 999023273)
    checkHiveHashForIntervalType("interval -999 millisecond", -998976727)

    // ----- SECOND -----

    // basic case
    checkHiveHashForIntervalType("interval 1 second", 23310)

    // negative
    checkHiveHashForIntervalType("interval -1 second", 23273)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 second", 23273)
    checkHiveHashForIntervalType("interval 2147483647 second", -2147460412)
    checkHiveHashForIntervalType("interval -2147483648 second", -2147460412)

    // Out of range for both Hive and Spark
    // Hive throws an exception. Spark overflows and returns wrong output
    // checkHiveHashForIntervalType("interval 9999999999 second", 0)

    // ----- MINUTE -----

    // basic cases
    checkHiveHashForIntervalType("interval 1 minute", 25493)

    // negative
    checkHiveHashForIntervalType("interval -1 minute", 25456)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 minute", 23273)
    checkHiveHashForIntervalType("interval 2147483647 minute", 21830)
    checkHiveHashForIntervalType("interval -2147483648 minute", 22163)

    // Out of range for both Hive and Spark
    // Hive throws an exception. Spark overflows and returns wrong output
    // checkHiveHashForIntervalType("interval 9999999999 minute", 0)

    // ----- HOUR -----

    // basic case
    checkHiveHashForIntervalType("interval 1 hour", 156473)

    // negative
    checkHiveHashForIntervalType("interval -1 hour", 156436)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 hour", 23273)
    checkHiveHashForIntervalType("interval 2147483647 hour", -62308)
    checkHiveHashForIntervalType("interval -2147483648 hour", -43327)

    // Out of range for both Hive and Spark
    // Hive throws an exception. Spark overflows and returns wrong output
    // checkHiveHashForIntervalType("interval 9999999999 hour", 0)

    // ----- DAY -----

    // basic cases
    checkHiveHashForIntervalType("interval 1 day", 3220073)

    // negative
    checkHiveHashForIntervalType("interval -1 day", 3220036)

    // edge / boundary cases
    checkHiveHashForIntervalType("interval 0 day", 23273)
    checkHiveHashForIntervalType("interval 106751991 day", -451506760)
    checkHiveHashForIntervalType("interval -106751991 day", -451514123)

    // Hive supports `day` for a longer range but Spark's range is smaller
    // The check for range is done at the parser level so this does not fail in Spark
    // checkHiveHashForIntervalType("interval -2147483648 day", -1575127)
    // checkHiveHashForIntervalType("interval 2147483647 day", -4767228)

    // Out of range for both Hive and Spark
    // Hive throws an exception. Spark overflows and returns wrong output
    // checkHiveHashForIntervalType("interval 9999999999 day", 0)

    // ----- MIX -----

    checkHiveHashForIntervalType("interval 0 day 0 hour", 23273)
    checkHiveHashForIntervalType("interval 0 day 0 hour 0 minute", 23273)
    checkHiveHashForIntervalType("interval 0 day 0 hour 0 minute 0 second", 23273)
    checkHiveHashForIntervalType("interval 0 day 0 hour 0 minute 0 second 0 millisecond", 23273)
    checkHiveHashForIntervalType(
      "interval 0 day 0 hour 0 minute 0 second 0 millisecond 0 microsecond", 23273)

    checkHiveHashForIntervalType("interval 6 day 15 hour", 21202073)
    checkHiveHashForIntervalType("interval 5 day 4 hour 8 minute", 16557833)
    checkHiveHashForIntervalType("interval -23 day 56 hour -1111113 minute 9898989 second",
      -2128468593)
    checkHiveHashForIntervalType("interval 66 day 12 hour 39 minute 23 second 987 millisecond",
      1199697904)
    checkHiveHashForIntervalType(
      "interval 66 day 12 hour 39 minute 23 second 987 millisecond 123 microsecond", 1199820904)
  }

  test("hive-hash for array") {
    // empty array
    checkHiveHash(
      input = new GenericArrayData(Array[Int]()),
      dataType = ArrayType(IntegerType, containsNull = false),
      expected = 0)

    // basic case
    checkHiveHash(
      input = new GenericArrayData(Array(1, 10000, Int.MaxValue)),
      dataType = ArrayType(IntegerType, containsNull = false),
      expected = -2147172688L)

    // with negative values
    checkHiveHash(
      input = new GenericArrayData(Array(-1L, 0L, 999L, Int.MinValue.toLong)),
      dataType = ArrayType(LongType, containsNull = false),
      expected = -2147452680L)

    // with nulls only
    val arrayTypeWithNull = ArrayType(IntegerType, containsNull = true)
    checkHiveHash(
      input = new GenericArrayData(Array(null, null)),
      dataType = arrayTypeWithNull,
      expected = 0)

    // mix with null
    checkHiveHash(
      input = new GenericArrayData(Array(-12221, 89, null, 767)),
      dataType = arrayTypeWithNull,
      expected = -363989515)

    // nested with array
    checkHiveHash(
      input = new GenericArrayData(
        Array(
          new GenericArrayData(Array(1234L, -9L, 67L)),
          new GenericArrayData(Array(null, null)),
          new GenericArrayData(Array(55L, -100L, -2147452680L))
        )),
      dataType = ArrayType(ArrayType(LongType)),
      expected = -1007531064)

    // nested with map
    checkHiveHash(
      input = new GenericArrayData(
        Array(
          new ArrayBasedMapData(
            new GenericArrayData(Array(-99, 1234)),
            new GenericArrayData(Array(UTF8String.fromString("sql"), null))),
          new ArrayBasedMapData(
            new GenericArrayData(Array(67)),
            new GenericArrayData(Array(UTF8String.fromString("apache spark"))))
        )),
      dataType = ArrayType(MapType(IntegerType, StringType)),
      expected = 1139205955)
  }

  test("hive-hash for map") {
    val mapType = MapType(IntegerType, StringType)

    // empty map
    checkHiveHash(
      input = new ArrayBasedMapData(new GenericArrayData(Array()), new GenericArrayData(Array())),
      dataType = mapType,
      expected = 0)

    // basic case
    checkHiveHash(
      input = new ArrayBasedMapData(
        new GenericArrayData(Array(1, 2)),
        new GenericArrayData(Array(UTF8String.fromString("foo"), UTF8String.fromString("bar")))),
      dataType = mapType,
      expected = 198872)

    // with null value
    checkHiveHash(
      input = new ArrayBasedMapData(
        new GenericArrayData(Array(55, -99)),
        new GenericArrayData(Array(UTF8String.fromString("apache spark"), null))),
      dataType = mapType,
      expected = 1142704473)

    // nesting (only values can be nested as keys have to be primitive datatype)
    val nestedMapType = MapType(IntegerType, MapType(IntegerType, StringType))
    checkHiveHash(
      input = new ArrayBasedMapData(
        new GenericArrayData(Array(1, -100)),
        new GenericArrayData(
          Array(
            new ArrayBasedMapData(
              new GenericArrayData(Array(-99, 1234)),
              new GenericArrayData(Array(UTF8String.fromString("sql"), null))),
            new ArrayBasedMapData(
              new GenericArrayData(Array(67)),
              new GenericArrayData(Array(UTF8String.fromString("apache spark"))))
          ))),
      dataType = nestedMapType,
      expected = -1142817416)
  }

  test("hive-hash for struct") {
    // basic
    val row = new GenericInternalRow(Array[Any](1, 2, 3))
    checkHiveHash(
      input = row,
      dataType =
        new StructType()
          .add("col1", IntegerType)
          .add("col2", IntegerType)
          .add("col3", IntegerType),
      expected = 1026)

    // mix of several datatypes
    val structType = new StructType()
      .add("null", NullType)
      .add("boolean", BooleanType)
      .add("byte", ByteType)
      .add("short", ShortType)
      .add("int", IntegerType)
      .add("long", LongType)
      .add("arrayOfString", arrayOfString)
      .add("mapOfString", mapOfString)

    val rowValues = new ArrayBuffer[Any]()
    rowValues += null
    rowValues += true
    rowValues += 1
    rowValues += 2
    rowValues += Int.MaxValue
    rowValues += Long.MinValue
    rowValues += new GenericArrayData(Array(
      UTF8String.fromString("apache spark"),
      UTF8String.fromString("hello world")
    ))
    rowValues += new ArrayBasedMapData(
      new GenericArrayData(Array(UTF8String.fromString("project"), UTF8String.fromString("meta"))),
      new GenericArrayData(Array(UTF8String.fromString("apache spark"), null))
    )

    val row2 = new GenericInternalRow(rowValues.toArray)
    checkHiveHash(
      input = row2,
      dataType = structType,
      expected = -2119012447)
  }

  private val structOfString = new StructType().add("str", StringType)
  private val structOfUDT = new StructType().add("udt", new ExamplePointUDT, false)
  private val arrayOfString = ArrayType(StringType)
  private val arrayOfNull = ArrayType(NullType)
  private val mapOfString = MapType(StringType, StringType)
  private val arrayOfUDT = ArrayType(new ExamplePointUDT, false)

  private val primitiveSchema = new StructType()
    .add("null", NullType)
    .add("boolean", BooleanType)
    .add("byte", ByteType)
    .add("short", ShortType)
    .add("int", IntegerType)
    .add("long", LongType)
    .add("float", FloatType)
    .add("double", DoubleType)
    .add("bigDecimal", DecimalType.SYSTEM_DEFAULT)
    .add("smallDecimal", DecimalType.USER_DEFAULT)
    .add("string", StringType)
    .add("binary", BinaryType)
    .add("date", DateType)
    .add("timestamp", TimestampType)
    .add("udt", new ExamplePointUDT)

  private val arraySchema = new StructType()
    .add("arrayOfNull", arrayOfNull)
    .add("arrayOfString", arrayOfString)
    .add("arrayOfArrayOfString", ArrayType(arrayOfString))
    .add("arrayOfArrayOfInt", ArrayType(ArrayType(IntegerType)))
    .add("arrayOfStruct", ArrayType(structOfString))
    .add("arrayOfUDT", arrayOfUDT)

  private val structSchema = new StructType()
    .add("structOfString", structOfString)
    .add("structOfStructOfString", new StructType().add("struct", structOfString))
    .add("structOfArray", new StructType().add("array", arrayOfString))
    .add("structOfUDT", structOfUDT)

  testHash(primitiveSchema)
  testHash(arraySchema)
  testHash(structSchema)

  testHash128(primitiveSchema)
  testHash128(arraySchema)
  testHash128(structSchema)

  test("hive-hash for decimal") {
    def checkHiveHashForDecimal(
        input: String,
        precision: Int,
        scale: Int,
        expected: Long): Unit = {
      val decimalType = DataTypes.createDecimalType(precision, scale)
      val decimal = {
        val value = Decimal.apply(new java.math.BigDecimal(input))
        if (value.changePrecision(precision, scale)) value else null
      }

      checkHiveHash(decimal, decimalType, expected)
    }

    checkHiveHashForDecimal("18", 38, 0, 558)
    checkHiveHashForDecimal("-18", 38, 0, -558)
    checkHiveHashForDecimal("-18", 38, 12, -558)
    checkHiveHashForDecimal("18446744073709001000", 38, 19, 0)
    checkHiveHashForDecimal("-18446744073709001000", 38, 22, 0)
    checkHiveHashForDecimal("-18446744073709001000", 38, 3, 17070057)
    checkHiveHashForDecimal("18446744073709001000", 38, 4, -17070057)
    checkHiveHashForDecimal("9223372036854775807", 38, 4, 2147482656)
    checkHiveHashForDecimal("-9223372036854775807", 38, 5, -2147482656)
    checkHiveHashForDecimal("00000.00000000000", 38, 34, 0)
    checkHiveHashForDecimal("-00000.00000000000", 38, 11, 0)
    checkHiveHashForDecimal("123456.1234567890", 38, 2, 382713974)
    checkHiveHashForDecimal("123456.1234567890", 38, 20, 1871500252)
    checkHiveHashForDecimal("123456.1234567890", 38, 10, 1871500252)
    checkHiveHashForDecimal("-123456.1234567890", 38, 10, -1871500234)
    checkHiveHashForDecimal("123456.1234567890", 38, 0, 3827136)
    checkHiveHashForDecimal("-123456.1234567890", 38, 0, -3827136)
    checkHiveHashForDecimal("123456.1234567890", 38, 20, 1871500252)
    checkHiveHashForDecimal("-123456.1234567890", 38, 20, -1871500234)
    checkHiveHashForDecimal("123456.123456789012345678901234567890", 38, 0, 3827136)
    checkHiveHashForDecimal("-123456.123456789012345678901234567890", 38, 0, -3827136)
    checkHiveHashForDecimal("123456.123456789012345678901234567890", 38, 10, 1871500252)
    checkHiveHashForDecimal("-123456.123456789012345678901234567890", 38, 10, -1871500234)
    checkHiveHashForDecimal("123456.123456789012345678901234567890", 38, 20, 236317582)
    checkHiveHashForDecimal("-123456.123456789012345678901234567890", 38, 20, -236317544)
    checkHiveHashForDecimal("123456.123456789012345678901234567890", 38, 30, 1728235666)
    checkHiveHashForDecimal("-123456.123456789012345678901234567890", 38, 30, -1728235608)
    checkHiveHashForDecimal("123456.123456789012345678901234567890", 38, 31, 1728235666)
  }

  for (collation <- Seq("UTF8_LCASE", "UNICODE_CI", "UTF8_BINARY")) {
    test(s"hash check for collated $collation strings - collation aware") {
      val s1 = "aaa"
      val s2 = "AAA"

      val murmur3Hash1 = CollationAwareMurmur3Hash(
        Seq(Collate(Literal(s1), ResolvedCollation(collation))),
        42
      )
      val murmur3Hash2 = CollationAwareMurmur3Hash(
        Seq(Collate(Literal(s2), ResolvedCollation(collation))),
        42
      )

      // Interpreted hash values for s1 and s2
      val interpretedHash1 = murmur3Hash1.eval()
      val interpretedHash2 = murmur3Hash2.eval()

      // Check that interpreted and codegen hashes are equal
      checkEvaluation(murmur3Hash1, interpretedHash1)
      checkEvaluation(murmur3Hash2, interpretedHash2)

      if (CollationFactory.fetchCollation(collation).isUtf8BinaryType) {
        assert(interpretedHash1 != interpretedHash2)
      } else {
        assert(interpretedHash1 == interpretedHash2)
      }
    }
  }

  for (collation <- Seq("UTF8_LCASE", "UNICODE_CI", "UTF8_BINARY")) {
    test(s"hash check for collated $collation strings - collation agnostic") {
      val s1 = "aaa"
      val s2 = "AAA"

      val murmur3Hash1 = Murmur3Hash(Seq(Collate(Literal(s1), ResolvedCollation(collation))), 42)
      val murmur3Hash2 = Murmur3Hash(Seq(Collate(Literal(s2), ResolvedCollation(collation))), 42)

      // Interpreted hash values for s1 and s2
      val interpretedHash1 = murmur3Hash1.eval()
      val interpretedHash2 = murmur3Hash2.eval()

      // Check that interpreted and codegen hashes are equal
      checkEvaluation(murmur3Hash1, interpretedHash1)
      checkEvaluation(murmur3Hash2, interpretedHash2)

      assert(interpretedHash1 != interpretedHash2)

      // Check that the hash computed is the same as the UTF8_BINARY version of it.
      if (!CollationFactory.fetchCollation(collation).isUtf8BinaryType) {
        Seq[String](s1, s2).foreach { s =>
          val utf8BinaryStringExpr = Collate(Literal(s), ResolvedCollation("UTF8_BINARY"))
          val murmur3HashBinary = Murmur3Hash(Seq(utf8BinaryStringExpr), 42)
          val hashBinary = murmur3HashBinary.eval()
          val murmur3Hash = Murmur3Hash(Seq(Collate(Literal(s), ResolvedCollation(collation))), 42)
          val interpretedHash = murmur3Hash.eval()
          assert(interpretedHash == hashBinary)
        }
      }
    }
  }

  // Below we test the `Murmur3Hash` and `XxHash64` expressions for the old behavior before the fix.
  // The expected values have been computed using the old implementation of the expression.
  test("SPARK-52828: always collation aware hash expression") {
    withSQLConf(SQLConf.COLLATION_AWARE_HASHING_ENABLED.key -> "true") {
      val testCases = Seq[(String, String, Int, Long)](
        // UTF8_BINARY
        ("AAA", "UTF8_BINARY", 22125783, 3965631622972380050L),
        ("AAA  ", "UTF8_BINARY", 399014599, 196039582279068044L),
        ("aaa", "UTF8_BINARY", -1689629761, 2465751751477118478L),
        ("aaa   ", "UTF8_BINARY", -1721438718, -2249763606958050730L),
        // UTF8_BINARY_RTRIM
        ("AAA", "UTF8_BINARY_RTRIM", -1493064582, 982928955165138586L),
        ("AAA  ", "UTF8_BINARY_RTRIM", -1493064582, 982928955165138586L),
        ("aaa", "UTF8_BINARY_RTRIM", 2132077201, -4940759280126763524L),
        ("aaa   ", "UTF8_BINARY_RTRIM", 2132077201, -4940759280126763524L),
        // UTF8_LCASE
        ("AAA", "UTF8_LCASE", 2132077201, -4940759280126763524L),
        ("AAA  ", "UTF8_LCASE", -619073595, -1146641051608991690L),
        ("aaa", "UTF8_LCASE", 2132077201, -4940759280126763524L),
        ("aaa   ", "UTF8_LCASE", -1498994355, -739345240752106297L),
        // UTF8_LCASE_RTRIM
        ("AAA", "UTF8_LCASE_RTRIM", 2132077201, -4940759280126763524L),
        ("AAA  ", "UTF8_LCASE_RTRIM", 2132077201, -4940759280126763524L),
        ("aaa", "UTF8_LCASE_RTRIM", 2132077201, -4940759280126763524L),
        ("aaa   ", "UTF8_LCASE_RTRIM", 2132077201, -4940759280126763524L),
        // UNICODE
        ("AAA", "UNICODE", 128537619, 49663227161197117L),
        ("AAA  ", "UNICODE", 82814175, 3618364417906061797L),
        ("aaa", "UNICODE", -1822783942, 290910714161494507L),
        ("aaa   ", "UNICODE", -896289340, 1025563887784400925L),
        // UNICODE_RTRIM
        ("AAA", "UNICODE_RTRIM", 128537619, 49663227161197117L),
        ("AAA  ", "UNICODE_RTRIM", 128537619, 49663227161197117L),
        ("aaa", "UNICODE_RTRIM", -1822783942, 290910714161494507L),
        ("aaa   ", "UNICODE_RTRIM", -1822783942, 290910714161494507L),
        // UNICODE_CI
        ("AAA", "UNICODE_CI", -443043098, -6629915645815515868L),
        ("AAA  ", "UNICODE_CI", 667473856, -3263604567598338200L),
        ("aaa", "UNICODE_CI", -443043098, -6629915645815515868L),
        ("aaa   ", "UNICODE_CI", -390983808, -5159733933636691741L),
        // UNICODE_CI_RTRIM
        ("AAA", "UNICODE_CI_RTRIM", -443043098, -6629915645815515868L),
        ("AAA  ", "UNICODE_CI_RTRIM", -443043098, -6629915645815515868L),
        ("aaa", "UNICODE_CI_RTRIM", -443043098, -6629915645815515868L),
        ("aaa   ", "UNICODE_CI_RTRIM", -443043098, -6629915645815515868L)
      )
      testCases.foreach { case (str, collationName, expectedMurmur3, expectedXxHash64) =>
        val stringExpr = Collate(Literal(str), ResolvedCollation(collationName))
        val murmur3Expr = Murmur3Hash(Seq(stringExpr), 42)
        checkEvaluation(murmur3Expr, expectedMurmur3)
        val xxHash64Expr = XxHash64(Seq(stringExpr), 42L)
        checkEvaluation(xxHash64Expr, expectedXxHash64)
      }
    }
  }

  test("SPARK-52828: backward-compatible hash API should reject UTF8_LCASE collation") {
    // This test verifies that the legacy hash API throws an exception when used with
    // collation-aware strings such as UTF8_LCASE. The assertion ensures we catch unsupported
    // usage early via the internal assertion (SchemaUtils.hasNonUTF8BinaryCollation).
    val expr_lcase = Collate(Literal("AAA"), ResolvedCollation("UTF8_LCASE"))
    intercept[IllegalArgumentException] {
      Murmur3HashFunction.hash(expr_lcase.eval(null), expr_lcase.dataType, 42)
    }
    intercept[IllegalArgumentException] {
      XxHash64Function.hash(expr_lcase.eval(null), expr_lcase.dataType, 42)
    }
    intercept[IllegalArgumentException] {
      HiveHashFunction.hash(expr_lcase.eval(null), expr_lcase.dataType, 42)
    }

    val expr_utf8bin = Collate(Literal("AAA"), ResolvedCollation("UTF8_BINARY"))
    Murmur3HashFunction.hash(expr_utf8bin.eval(null), expr_utf8bin.dataType, 42)
    XxHash64Function.hash(expr_utf8bin.eval(null), expr_utf8bin.dataType, 42)
    HiveHashFunction.hash(expr_utf8bin.eval(null), expr_utf8bin.dataType, 42)
  }

  test("SPARK-18207: Compute hash for a lot of expressions") {
    def checkResult(schema: StructType, input: InternalRow): Unit = {
      val exprs = schema.fields.zipWithIndex.map { case (f, i) =>
        BoundReference(i, f.dataType, true)
      }.toImmutableArraySeq
      val murmur3HashExpr = Murmur3Hash(exprs, 42)
      val murmur3HashPlan = GenerateMutableProjection.generate(Seq(murmur3HashExpr))
      val murmursHashEval = Murmur3Hash(exprs, 42).eval(input)
      assert(murmur3HashPlan(input).getInt(0) == murmursHashEval)

      val xxHash64Expr = XxHash64(exprs, 42)
      val xxHash64Plan = GenerateMutableProjection.generate(Seq(xxHash64Expr))
      val xxHash64Eval = XxHash64(exprs, 42).eval(input)
      assert(xxHash64Plan(input).getLong(0) == xxHash64Eval)

      val xxHash3Expr = XxHash3(exprs, 42L)
      val xxHash3Plan = GenerateMutableProjection.generate(Seq(xxHash3Expr))
      val xxHash3Eval = XxHash3(exprs, 42L).eval(input)
      assert(xxHash3Plan(input).getLong(0) == xxHash3Eval)

      val hiveHashExpr = HiveHash(exprs)
      val hiveHashPlan = GenerateMutableProjection.generate(Seq(hiveHashExpr))
      val hiveHashEval = HiveHash(exprs).eval(input)
      assert(hiveHashPlan(input).getInt(0) == hiveHashEval)
    }

    val N = 1000
    val wideRow = new GenericInternalRow(
      Seq.tabulate(N)(i => UTF8String.fromString(i.toString)).toArray[Any])
    val schema = StructType((1 to N).map(i => StructField(i.toString, StringType)))
    checkResult(schema, wideRow)

    val nestedRow = InternalRow(wideRow)
    val nestedSchema = new StructType().add("nested", schema)
    checkResult(nestedSchema, nestedRow)
  }

  test("SPARK-22284: Compute hash for nested structs") {
    val M = 80
    val N = 10
    val L = M * N
    val O = 50
    val seed = 42

    val wideRow = new GenericInternalRow(Seq.tabulate(O)(k =>
      new GenericInternalRow(Seq.tabulate(M)(j =>
        new GenericInternalRow(Seq.tabulate(N)(i =>
          new GenericInternalRow(Array[Any](
            UTF8String.fromString((k * L + j * N + i).toString))))
          .toArray[Any])).toArray[Any])).toArray[Any])
    val inner = new StructType(
      (0 until N).map(_ => StructField("structOfString", structOfString)).toArray)
    val outer = new StructType(
      (0 until M).map(_ => StructField("structOfStructOfString", inner)).toArray)
    val schema = new StructType(
      (0 until O).map(_ => StructField("structOfStructOfStructOfString", outer)).toArray)
    val exprs = schema.fields.zipWithIndex.map { case (f, i) =>
      BoundReference(i, f.dataType, true)
    }.toImmutableArraySeq
    val murmur3HashExpr = Murmur3Hash(exprs, 42)
    val murmur3HashPlan = GenerateMutableProjection.generate(Seq(murmur3HashExpr))

    val murmursHashEval = Murmur3Hash(exprs, 42).eval(wideRow)
    assert(murmur3HashPlan(wideRow).getInt(0) == murmursHashEval)
  }

  test("SPARK-30633: xxHash with different type seeds") {
    val literal = Literal.create(42L, LongType)

    val longSeeds = Seq(
      Long.MinValue,
      Integer.MIN_VALUE.toLong - 1L,
      0L,
      Integer.MAX_VALUE.toLong + 1L,
      Long.MaxValue
    )
    for (seed <- longSeeds) {
      checkEvaluation(XxHash64(Seq(literal), seed), XxHash64(Seq(literal), seed).eval())
    }

    val intSeeds = Seq(
      Integer.MIN_VALUE,
      0,
      Integer.MAX_VALUE
    )
    for (seed <- intSeeds) {
      checkEvaluation(XxHash64(Seq(literal), seed), XxHash64(Seq(literal), seed).eval())
    }

    checkEvaluation(XxHash64(Seq(literal), 100), XxHash64(Seq(literal), 100L).eval())
    checkEvaluation(XxHash64(Seq(literal), 100L), XxHash64(Seq(literal), 100).eval())
  }

  test("SPARK-35113: HashExpression support DayTimeIntervalType/YearMonthIntervalType") {
    val dayTime = Literal.create(Duration.ofSeconds(1237123123), DayTimeIntervalType())
    val yearMonth = Literal.create(Period.ofMonths(1234), YearMonthIntervalType())
    checkEvaluation(Murmur3Hash(Seq(dayTime), 10), -428664612)
    checkEvaluation(Murmur3Hash(Seq(yearMonth), 10), -686520021)
    checkEvaluation(XxHash64(Seq(dayTime), 10), 8228802290839366895L)
    checkEvaluation(XxHash64(Seq(yearMonth), 10), -1774215319882784110L)
    checkEvaluation(XxHash3(Seq(dayTime), 10L), XxHash3(Seq(dayTime), 10L).eval())
    checkEvaluation(XxHash3(Seq(yearMonth), 10L), XxHash3(Seq(yearMonth), 10L).eval())
    checkEvaluation(HiveHash(Seq(dayTime)), 743331816)
    checkEvaluation(HiveHash(Seq(yearMonth)), 1234)
  }

  test("SPARK-35207: Compute hash consistent between -0.0 and 0.0") {
    def checkResult(exprs1: Expression, exprs2: Expression): Unit = {
      checkEvaluation(Murmur3Hash(Seq(exprs1), 42), Murmur3Hash(Seq(exprs2), 42).eval())
      checkEvaluation(XxHash64(Seq(exprs1), 42), XxHash64(Seq(exprs2), 42).eval())
      checkEvaluation(XxHash3(Seq(exprs1), 42L), XxHash3(Seq(exprs2), 42L).eval())
      checkEvaluation(HiveHash(Seq(exprs1)), HiveHash(Seq(exprs2)).eval())
    }

    checkResult(Literal.create(-0D, DoubleType), Literal.create(0D, DoubleType))
    checkResult(Literal.create(-0F, FloatType), Literal.create(0F, FloatType))
  }

  test("Support TimeType") {
    val time = Literal.create(LocalTime.of(23, 50, 59, 123456000), TimeType())
    checkEvaluation(Murmur3Hash(Seq(time), 10), 545499634)
    checkEvaluation(XxHash64(Seq(time), 10), -3550518982366774761L)
    checkEvaluation(XxHash3(Seq(time), 10L), XxHash3(Seq(time), 10L).eval())
    checkEvaluation(HiveHash(Seq(time)), -1567775210)
  }

  test("HashExpression supports nanosecond timestamp types") {
    // (epochMicros, nanosWithinMicro) pairs covering zero/mid/max nanos, negative micros, and
    // the Long epoch-micro boundaries.
    val values = Seq(
      TimestampNanosVal.fromParts(0L, 0.toShort),
      TimestampNanosVal.fromParts(1L, 1.toShort),
      TimestampNanosVal.fromParts(1234567890L, 999.toShort),
      TimestampNanosVal.fromParts(-1L, 500.toShort),
      TimestampNanosVal.fromParts(Long.MinValue, 0.toShort),
      TimestampNanosVal.fromParts(Long.MaxValue, 999.toShort))

    Seq(TimestampNTZNanosType(9), TimestampLTZNanosType(9),
        TimestampNTZNanosType(7), TimestampLTZNanosType(7)).foreach { dt =>
      (values :+ null).foreach { v =>
        // 1) Literal child: the value is embedded as a constant, so this asserts that the
        // interpreted and codegen paths agree. (The unsafe projection here only round-trips the
        // scalar hash result, not the nanos input -- that path is covered below.)
        val lit = Literal.create(v, dt)
        checkEvaluation(Murmur3Hash(Seq(lit), 42), Murmur3Hash(Seq(lit), 42).eval())
        checkEvaluation(XxHash64(Seq(lit), 42L), XxHash64(Seq(lit), 42L).eval())
        checkEvaluation(HiveHash(Seq(lit)), HiveHash(Seq(lit)).eval())

        // 2) BoundReference over a row: drives the ordinal row-read (getTimestampNTZNanos /
        // getTimestampLTZNanos) and the UnsafeRow round-trip of the nanos value itself -- the
        // real GROUP BY / shuffle / join input path that the literal case above skips.
        val row = InternalRow(v)
        val ref = BoundReference(0, dt, nullable = true)
        checkEvaluation(Murmur3Hash(Seq(ref), 42), Murmur3Hash(Seq(ref), 42).eval(row), row)
        checkEvaluation(XxHash64(Seq(ref), 42L), XxHash64(Seq(ref), 42L).eval(row), row)
        checkEvaluation(HiveHash(Seq(ref)), HiveHash(Seq(ref)).eval(row), row)
      }
    }
  }

  test("nanosecond timestamp hash is consistent with equality") {
    val dt = TimestampNTZNanosType(9)
    def lit(micros: Long, nanos: Short): Literal =
      Literal.create(TimestampNanosVal.fromParts(micros, nanos), dt)

    val a = lit(1234567890L, 123)
    val aCopy = lit(1234567890L, 123)
    val diffNanos = lit(1234567890L, 124) // same micros, different sub-micro nanos
    val diffMicros = lit(1234567891L, 123) // different micros, same nanos

    Seq[Expression => Any](
      e => Murmur3Hash(Seq(e), 42).eval(),
      e => XxHash64(Seq(e), 42L).eval(),
      e => HiveHash(Seq(e)).eval()).foreach { hash =>
      // Equal values hash equally.
      assert(hash(a) === hash(aCopy))
      // Both fields contribute to the hash (guards against a dropped epochMicros/nanos field).
      assert(hash(a) !== hash(diffNanos))
      assert(hash(a) !== hash(diffMicros))
    }
  }

  test("nanosecond timestamp hash matches expected golden values") {
    // The expected values are composed independently of the expression under test -- directly
    // from the primitive hashers (and the separate hashTimestamp for Hive) with an explicit
    // epochMicros-then-nanosWithinMicro folding order. So a wrong seed/constant or a swapped
    // field order in the dispatch is caught, rather than masked by comparing the expression
    // against itself.
    val micros = 1234567890L
    val nanos: Short = 789
    val v = TimestampNanosVal.fromParts(micros, nanos)
    val seed = 42
    Seq(TimestampNTZNanosType(9), TimestampLTZNanosType(9)).foreach { dt =>
      val lit = Literal.create(v, dt)
      checkEvaluation(
        Murmur3Hash(Seq(lit), seed),
        Murmur3_x86_32.hashInt(nanos, Murmur3_x86_32.hashLong(micros, seed)))
      checkEvaluation(
        XxHash64(Seq(lit), seed.toLong),
        XXH64.hashInt(nanos, XXH64.hashLong(micros, seed.toLong)))
      checkEvaluation(
        HiveHash(Seq(lit)),
        ((HiveHashFunction.hashTimestamp(micros) * 37) + nanos).toInt)
    }
  }

  private def testHash(inputSchema: StructType): Unit = {
    val inputGenerator = RandomDataGenerator.forType(inputSchema, nullable = false).get
    val toRow = ExpressionEncoder(inputSchema).createSerializer()
    val seed = scala.util.Random.nextInt()
    test(s"murmur3/xxHash64/xxHash3/hive hash: ${inputSchema.simpleString}") {
      for (_ <- 1 to 10) {
        val input = toRow(inputGenerator.apply().asInstanceOf[Row]).asInstanceOf[UnsafeRow]
        val literals = input.toSeq(inputSchema).zip(inputSchema.map(_.dataType)).map {
          case (value, dt) => Literal.create(value, dt)
        }
        // Only test the interpreted version has same result with codegen version.
        checkEvaluation(Murmur3Hash(literals, seed), Murmur3Hash(literals, seed).eval())
        checkEvaluation(XxHash64(literals, seed), XxHash64(literals, seed).eval())
        checkEvaluation(XxHash3(literals, seed.toLong), XxHash3(literals, seed.toLong).eval())
        checkEvaluation(HiveHash(literals), HiveHash(literals).eval())
      }
    }

    val longSeed = Math.abs(seed).toLong + Integer.MAX_VALUE.toLong
    test(s"SPARK-30633: xxHash64 with long seed: ${inputSchema.simpleString}") {
      for (_ <- 1 to 10) {
        val input = toRow(inputGenerator.apply().asInstanceOf[Row]).asInstanceOf[UnsafeRow]
        val literals = input.toSeq(inputSchema).zip(inputSchema.map(_.dataType)).map {
          case (value, dt) => Literal.create(value, dt)
        }
        // Only test the interpreted version has same result with codegen version.
        checkEvaluation(XxHash64(literals, longSeed), XxHash64(literals, longSeed).eval())
        checkEvaluation(XxHash3(literals, longSeed), XxHash3(literals, longSeed).eval())
      }
    }
  }

  private def testHash128(inputSchema: StructType): Unit = {
    val inputGenerator = RandomDataGenerator.forType(inputSchema, nullable = false).get
    val toRow = ExpressionEncoder(inputSchema).createSerializer()
    test(s"xxHash128: ${inputSchema.simpleString}") {
      for (_ <- 1 to 10) {
        val input = toRow(inputGenerator.apply().asInstanceOf[Row]).asInstanceOf[UnsafeRow]
        val literals = input.toSeq(inputSchema).zip(inputSchema.map(_.dataType)).map {
          case (value, dt) => Literal.create(value, dt)
        }
        // Interpreted and codegen paths must agree.
        checkEvaluation(new XxHash128(literals), new XxHash128(literals).eval())
      }
    }
  }
}
