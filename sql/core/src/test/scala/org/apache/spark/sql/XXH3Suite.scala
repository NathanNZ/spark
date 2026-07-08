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
import org.apache.spark.sql.functions.{lit, xxhash128, xxhash3}
import org.apache.spark.sql.test.SharedSparkSession

class XXH3Suite extends QueryTest with SharedSparkSession {

  import testImplicits._

  test("xxhash3: Scala API and SQL form return identical results") {
    val df = Seq(1 -> "a", 2 -> "b").toDF("i", "j")
    withTempView("tbl") {
      df.createOrReplaceTempView("tbl")
      checkAnswer(
        df.select(xxhash3($"i", $"j")),
        sql("SELECT xxhash3(i, j) FROM tbl"))
    }
  }

  test("xxhash128: Scala API and SQL form return identical results") {
    val df = Seq(1 -> "a", 2 -> "b").toDF("i", "j")
    withTempView("tbl") {
      df.createOrReplaceTempView("tbl")
      checkAnswer(
        df.select(xxhash128($"i", $"j")),
        sql("SELECT xxhash128(i, j) FROM tbl"))
    }
  }

  test("xxhash3(int) matches XXH3.hashInt64 at the algorithm boundary") {
    val expected = XXH3.hashInt64(1234, 0L)
    checkAnswer(
      spark.range(1).select(xxhash3(lit(1234))),
      Row(expected) :: Nil)
  }

  test("xxhash128(int) matches XXH3.hashInt128 at the algorithm boundary") {
    val expected = XXH3.hashInt128(1234, 0L)
    checkAnswer(
      spark.range(1).select(xxhash128(lit(1234))),
      Row(expected) :: Nil)
  }

  test("SPARK-45900: xxhash3 accepts MapType") {
    val df = spark.createDataset(Map("a" -> 1L, "b" -> 2L) :: Nil)
    val res = df.selectExpr("xxhash3(value)").collect()
    assert(res.length == 1)
  }

  test("SPARK-45900: xxhash128 accepts MapType") {
    val df = spark.createDataset(Map("a" -> 1L, "b" -> 2L) :: Nil)
    val res = df.selectExpr("xxhash128(value)").collect()
    assert(res.length == 1)
  }

  test("SPARK-45900: xxhash3 MapType: insertion-order independent") {
    val asc  = Map("alpha" -> 1L, "bravo" -> 2L, "charlie" -> 3L)
    val desc = Map("charlie" -> 3L, "bravo" -> 2L, "alpha" -> 1L)
    val rows = Seq(asc, desc).toDF("m").select(xxhash3($"m")).collect().map(_.getLong(0))
    assert(rows(0) == rows(1),
      s"xxhash3 should be insertion-order independent: got ${rows.mkString(", ")}")
  }

  test("SPARK-45900: xxhash128 MapType: insertion-order independent") {
    val asc  = Map("alpha" -> 1L, "bravo" -> 2L, "charlie" -> 3L)
    val desc = Map("charlie" -> 3L, "bravo" -> 2L, "alpha" -> 1L)
    val rows = Seq(asc, desc).toDF("m").select(xxhash128($"m")).collect().map(_.getAs[Array[Byte]](0))
    assert(java.util.Arrays.equals(rows(0), rows(1)), "xxhash128 should be insertion-order independent")
  }

  test("SPARK-45900: xxhash3 accepts VariantType") {
    val res = sql("""SELECT xxhash3(parse_json('{"a":1,"b":2}'))""").collect()
    assert(res.length == 1 && !res(0).isNullAt(0))
  }

  test("SPARK-45900: xxhash128 accepts VariantType") {
    val res = sql("""SELECT xxhash128(parse_json('{"a":1,"b":2}'))""").collect()
    assert(res.length == 1 && !res(0).isNullAt(0))
  }

  test("xxhash3 interior nulls perturb the result end-to-end") {
    val withInterior = spark.range(1)
      .select(xxhash3(lit(1), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    val withoutInterior = spark.range(1)
      .select(xxhash3(lit(1), lit(2)).as("h"))
      .head().getLong(0)
    assert(withInterior != withoutInterior,
      "interior null must change the hash")
  }

  test("xxhash3 null placement perturbs the result end-to-end") {
    val trailing = spark.range(1)
      .select(xxhash3(lit("Bob"), lit(1), lit(null)).as("h"))
      .head().getLong(0)
    val interior = spark.range(1)
      .select(xxhash3(lit("Bob"), lit(null), lit(1)).as("h"))
      .head().getLong(0)
    assert(trailing != interior,
      "same-width rows with different null placement must hash differently")
  }

  test("xxhash128 null placement perturbs the result end-to-end") {
    val trailing = spark.range(1)
      .select(xxhash128(lit("Bob"), lit(1), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val interior = spark.range(1)
      .select(xxhash128(lit("Bob"), lit(null), lit(1)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(trailing, interior),
      "same-width rows with different null placement must hash differently")
  }

  test("xxhash3 null run length perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxhash3(lit(1), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    val twoNulls = spark.range(1)
      .select(xxhash3(lit(1), lit(null), lit(null), lit(2)).as("h"))
      .head().getLong(0)
    assert(oneNull != twoNulls,
      "consecutive null runs with different lengths must hash differently")
  }

  test("xxhash128 null run length perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxhash128(lit(1), lit(null), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    val twoNulls = spark.range(1)
      .select(xxhash128(lit(1), lit(null), lit(null), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(oneNull, twoNulls),
      "consecutive null runs with different lengths must hash differently")
  }

  test("xxhash3 trailing nulls perturb the result end-to-end") {
    val noTrailing = spark.range(1)
      .select(xxhash3(lit(1), lit(2)).as("h"))
      .head().getLong(0)
    val oneTrailing = spark.range(1)
      .select(xxhash3(lit(1), lit(2), lit(null)).as("h"))
      .head().getLong(0)
    val twoTrailing = spark.range(1)
      .select(xxhash3(lit(1), lit(2), lit(null), lit(null)).as("h"))
      .head().getLong(0)
    assert(noTrailing != oneTrailing && oneTrailing != twoTrailing,
      "trailing nulls must change the hash")
  }

  test("xxhash128 trailing nulls perturb the result end-to-end") {
    val noTrailing = spark.range(1)
      .select(xxhash128(lit(1), lit(2)).as("h"))
      .head().getAs[Array[Byte]](0)
    val oneTrailing = spark.range(1)
      .select(xxhash128(lit(1), lit(2), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val twoTrailing = spark.range(1)
      .select(xxhash128(lit(1), lit(2), lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(noTrailing, oneTrailing) &&
      !java.util.Arrays.equals(oneTrailing, twoTrailing),
      "trailing nulls must change the hash")
  }

  test("xxhash3 all-null input width perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxhash3(lit(null)).as("h"))
      .head().getLong(0)
    val twoNulls = spark.range(1)
      .select(xxhash3(lit(null), lit(null)).as("h"))
      .head().getLong(0)
    val fourNulls = spark.range(1)
      .select(xxhash3(lit(null), lit(null), lit(null), lit(null)).as("h"))
      .head().getLong(0)
    assert(oneNull != twoNulls && twoNulls != fourNulls,
      "all-null rows with different widths must hash differently")
  }

  test("xxhash128 all-null input width perturbs the result end-to-end") {
    val oneNull = spark.range(1)
      .select(xxhash128(lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val twoNulls = spark.range(1)
      .select(xxhash128(lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    val fourNulls = spark.range(1)
      .select(xxhash128(lit(null), lit(null), lit(null), lit(null)).as("h"))
      .head().getAs[Array[Byte]](0)
    assert(!java.util.Arrays.equals(oneNull, twoNulls) &&
      !java.util.Arrays.equals(twoNulls, fourNulls),
      "all-null rows with different widths must hash differently")
  }

  test("xxhash3 with zero arguments throws AnalysisException") {
    intercept[AnalysisException] { Seq(1).toDF().selectExpr("xxhash3()") }
  }

  test("xxhash128 with zero arguments throws AnalysisException") {
    intercept[AnalysisException] { Seq(1).toDF().selectExpr("xxhash128()") }
  }
}
