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

package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.XXH3
import org.apache.spark.sql.functions.{lit, udf, xxh3_128, xxh3_128_hex, xxh3_64}
import org.apache.spark.sql.test.SharedSparkSession

class XXH3Suite extends QueryTest with SharedSparkSession {

  import testImplicits._

  test("xxh3_64: Scala API and SQL form return identical results") {
    val df = Seq(1 -> "a", 2 -> "b").toDF("i", "j")
    withTempView("tbl") {
      df.createOrReplaceTempView("tbl")
      checkAnswer(
        df.select(xxh3_64($"i", $"j")),
        sql("SELECT xxh3_64(i, j) FROM tbl"))
    }
  }

  test("xxh3_128: Scala API and SQL form return identical results") {
    val df = Seq(1 -> "a", 2 -> "b").toDF("i", "j")
    withTempView("tbl") {
      df.createOrReplaceTempView("tbl")
      checkAnswer(
        df.select(xxh3_128($"i", $"j")),
        sql("SELECT xxh3_128(i, j) FROM tbl"))
    }
  }

  test("xxh3_128_hex returns lowercase canonical hexadecimal output") {
    val df = Seq(("Spark", 1)).toDF("s", "i")
    checkAnswer(
      df.select(xxh3_128_hex($"s"), xxh3_128_hex($"s", $"i")),
      df.selectExpr(
        "lower(hex(xxh3_128(s)))",
        "lower(hex(xxh3_128(s, i)))"))
    assert(df.select(xxh3_128_hex($"s")).head().getString(0)
      .matches("[0-9a-f]{32}"))
  }

  test("xxh3_128_hex preserves unary and structural null semantics") {
    assert(spark.range(1).select(xxh3_128_hex(lit(null))).head().isNullAt(0))
    assert(spark.range(1).select(xxh3_128_hex(lit(null), lit(null)))
      .head().getString(0).matches("[0-9a-f]{32}"))
  }

  test("xxh3_64(binary) matches XXH3.hash64 at the algorithm boundary") {
    val input = Array[Byte](1, 2, 3, 4)
    val expected = XXH3.hash64(input)
    checkAnswer(
      spark.range(1).select(xxh3_64(lit(input))),
      Row(expected) :: Nil)
  }

  test("xxh3_128(binary) matches XXH3.hashBytes128 at the algorithm boundary") {
    val input = Array[Byte](1, 2, 3, 4)
    val expected = XXH3.hashBytes128(input, 0, input.length, 0L)
    checkAnswer(
      spark.range(1).select(xxh3_128(lit(input))),
      Row(expected) :: Nil)
  }

  test("SPARK-58677: xxh3_64 accepts MapType") {
    val df = spark.createDataset(Map("a" -> 1L, "b" -> 2L) :: Nil)
    val res = df.selectExpr("xxh3_64(value, 1)").collect()
    assert(res.length == 1)
  }

  test("SPARK-58677: xxh3_128 accepts MapType") {
    val df = spark.createDataset(Map("a" -> 1L, "b" -> 2L) :: Nil)
    val res = df.selectExpr("xxh3_128(value, 1)").collect()
    assert(res.length == 1)
  }

  test("SPARK-58677: unary complex types use structural hashing") {
    val df = Seq((Seq(1, 2), Map("a" -> 1), (3, "x"))).toDF("a", "m", "s")
    val result = df.select(
      xxh3_64($"a"), xxh3_128($"a"),
      xxh3_64($"m"), xxh3_128($"m"),
      xxh3_64($"s"), xxh3_128($"s")).head()

    assert(!result.isNullAt(0) && result.getAs[Array[Byte]](1).length == 16)
    assert(!result.isNullAt(2) && result.getAs[Array[Byte]](3).length == 16)
    assert(!result.isNullAt(4) && result.getAs[Array[Byte]](5).length == 16)
  }

  test("SPARK-58677: unary null complex values propagate null") {
    val result = spark.range(1).selectExpr(
      "xxh3_64(CAST(NULL AS ARRAY<INT>))",
      "xxh3_128(CAST(NULL AS MAP<STRING, INT>))",
      "xxh3_64(CAST(NULL AS STRUCT<a: INT>))").head()
    assert(result.toSeq.forall(_ == null))
  }

  test("SPARK-58677: unary structural arguments are evaluated once in join conditions") {
    val calls = sparkContext.longAccumulator("xxh3 unary join calls")
    val observed = udf((left: Long, right: Long) => {
      calls.add(1L)
      (left, right)
    })
    val left = spark.range(2).as("left")
    val right = spark.range(3).as("right")

    val result = left.join(right,
      xxh3_64(observed(left("id"), right("id"))).isNotNull).count()
    assert(result == 6L)
    assert(calls.value == 6L)
  }

  test("SPARK-58677: xxh3_64 MapType: insertion-order independent") {
    val asc  = Map("alpha" -> 1L, "bravo" -> 2L, "charlie" -> 3L)
    val desc = Map("charlie" -> 3L, "bravo" -> 2L, "alpha" -> 1L)
    val rows = Seq(asc, desc).toDF("m").select(xxh3_64($"m", lit(1)))
      .collect().map(_.getLong(0))
    assert(rows(0) == rows(1),
      s"xxh3_64 should be insertion-order independent: got ${rows.mkString(", ")}")
  }

  test("SPARK-58677: xxh3_128 MapType: insertion-order independent") {
    val asc  = Map("alpha" -> 1L, "bravo" -> 2L, "charlie" -> 3L)
    val desc = Map("charlie" -> 3L, "bravo" -> 2L, "alpha" -> 1L)
    val rows = Seq(asc, desc).toDF("m").select(xxh3_128($"m", lit(1)))
      .collect().map(_.getAs[Array[Byte]](0))
    assert(java.util.Arrays.equals(rows(0), rows(1)),
      "xxh3_128 should be insertion-order independent")
  }

  test("SPARK-58677: xxh3_64 accepts VariantType") {
    val res = sql("""SELECT xxh3_64(parse_json('{"a":1,"b":2}'), 1)""").collect()
    assert(res.length == 1 && !res(0).isNullAt(0))
  }

  test("SPARK-58677: xxh3_128 accepts VariantType") {
    val res = sql("""SELECT xxh3_128(parse_json('{"a":1,"b":2}'), 1)""").collect()
    assert(res.length == 1 && !res(0).isNullAt(0))
  }

  test("xxh3_64 interior nulls perturb the result end-to-end") {
    val withInterior = spark.range(1)
      .select(xxh3_64(lit(1), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    val withoutInterior = spark.range(1)
      .select(xxh3_64(lit(1), lit(2)).as("h"))
      .head().getLong(0)
    assert(withInterior != withoutInterior,
      "interior null must change the hash")
  }

  test("xxh3_64 null placement perturbs the result end-to-end") {
    val trailing = spark.range(1)
      .select(xxh3_64(lit("Bob"), lit(1), lit(null)).as("h"))
      .head().getLong(0)
    val interior = spark.range(1)
      .select(xxh3_64(lit("Bob"), lit(null), lit(1)).as("h"))
      .head().getLong(0)
    assert(trailing != interior,
      "same-width rows with different null placement must hash differently")
  }

  test("xxh3_128 null placement perturbs the result end-to-end") {
    val trailing = spark.range(1)
      .select(xxh3_128(lit("Bob"), lit(1), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val interior = spark.range(1)
      .select(xxh3_128(lit("Bob"), lit(null), lit(1)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(trailing, interior),
      "same-width rows with different null placement must hash differently")
  }

  test("xxh3_64 null run length perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxh3_64(lit(1), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    val twoNulls = spark.range(1)
      .select(xxh3_64(lit(1), lit(null), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    assert(oneNull != twoNulls,
      "consecutive null runs with different lengths must hash differently")
  }

  test("xxh3_128 null run length perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxh3_128(lit(1), lit(null), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    val twoNulls = spark.range(1)
      .select(xxh3_128(lit(1), lit(null), lit(null), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(oneNull, twoNulls),
      "consecutive null runs with different lengths must hash differently")
  }

  test("xxh3_64 trailing nulls perturb the result end-to-end") {
    val noTrailing = spark.range(1)
      .select(xxh3_64(lit(1), lit(2)).as("h"))
      .head().getLong(0)
    val oneTrailing = spark.range(1)
      .select(xxh3_64(lit(1), lit(2), lit(null)).as("h"))
      .head().getLong(0)
    val twoTrailing = spark.range(1)
      .select(xxh3_64(lit(1), lit(2), lit(null), lit(null)).as("h"))
      .head().getLong(0)
    assert(noTrailing != oneTrailing && oneTrailing != twoTrailing,
      "trailing nulls must change the hash")
  }

  test("xxh3_128 trailing nulls perturb the result end-to-end") {
    val noTrailing = spark.range(1)
      .select(xxh3_128(lit(1), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    val oneTrailing = spark.range(1)
      .select(xxh3_128(lit(1), lit(2), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val twoTrailing = spark.range(1)
      .select(xxh3_128(lit(1), lit(2), lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(noTrailing, oneTrailing) &&
      !java.util.Arrays.equals(oneTrailing, twoTrailing),
      "trailing nulls must change the hash")
  }

  test("xxh3_64 unary null propagates and wider all-null inputs are structural") {
    assert(spark.range(1).select(xxh3_64(lit(null)).as("h")).head().isNullAt(0))
    val twoNulls = spark.range(1)
      .select(xxh3_64(lit(null), lit(null)).as("h"))
      .head().getLong(0)
    val fourNulls = spark.range(1)
      .select(xxh3_64(lit(null), lit(null), lit(null), lit(null)).as("h"))
      .head().getLong(0)
    assert(twoNulls != fourNulls,
      "all-null rows with different widths must hash differently")
  }

  test("xxh3_128 unary null propagates and wider all-null inputs are structural") {
    assert(spark.range(1).select(xxh3_128(lit(null)).as("h")).head().isNullAt(0))
    val twoNulls = spark.range(1)
      .select(xxh3_128(lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val fourNulls = spark.range(1)
      .select(xxh3_128(lit(null), lit(null), lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(twoNulls, fourNulls),
      "all-null rows with different widths must hash differently")
  }

  test("xxh3_64 with zero arguments throws AnalysisException") {
    intercept[AnalysisException] { Seq(1).toDF().selectExpr("xxh3_64()") }
  }

  test("xxh3_128 with zero arguments throws AnalysisException") {
    intercept[AnalysisException] { Seq(1).toDF().selectExpr("xxh3_128()") }
  }
}
