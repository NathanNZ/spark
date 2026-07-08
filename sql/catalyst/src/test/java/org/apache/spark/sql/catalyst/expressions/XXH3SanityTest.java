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

/*
 * xxHash - Sanity test for the Spark XXH3 implementation
 * Verifies against the shared xxhash_test_vectors.json (generated from the C reference).
 * Port of xxhash_work/src/test/java/com/github/nathannz/xxhash64/Shared/XXH3SanityTest.java
 *
 * Copyright (C) 2012-2023 Yann Collet
 * Copyright (C) 2026 Nathan Holland
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 */
package org.apache.spark.sql.catalyst.expressions;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.spark.unsafe.Platform;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sanity tests for the Spark XXH3 implementation against the shared
 * xxhash_test_vectors.json (generated from the C reference implementation).
 *
 * <p>Uses the same pseudorandom sanity buffer as the C sanity_test.c:
 *   buffer[i] = (byte)(byteGen >>> 56);  byteGen *= PRIME64
 * where byteGen starts at PRIME32.
 */
@DisplayName("XXH3 Sanity Tests")
public class XXH3SanityTest {

  // Matches C implementation fillTestBuffer / sanity_test.c
  private static final long PRIME32 = 2654435761L;
  private static final long PRIME64 = 0x9E3779B185EBCA8DL;
  private static final int SANITY_BUFFER_SIZE = 4096 + 64 + 1;

  private static final int SECRET_OFFSET = 7;
  private static final int XXH3_SECRET_SIZE_MIN = 136;
  private static final int SECRET_SIZE = XXH3_SECRET_SIZE_MIN + 11;

  private static byte[] sanityBuffer;

  private static List<TestData64> xxh3_64BitsVectors;
  private static List<TestData128> xxh3_128BitsVectors;

  // -----------------------------------------------------------------------
  // Inline test-vector POJOs (mirrors TestVectorLoader in xxhash_work)
  // -----------------------------------------------------------------------

  static final class TestData64 {
    final int len;
    final long seed;
    final long expected;

    TestData64(int len, long seed, long expected) {
      this.len = len;
      this.seed = seed;
      this.expected = expected;
    }
  }

  static final class TestData128 {
    final int len;
    final long seed;
    final long expectedLow;
    final long expectedHigh;

    TestData128(int len, long seed, long expectedLow, long expectedHigh) {
      this.len = len;
      this.seed = seed;
      this.expectedLow = expectedLow;
      this.expectedHigh = expectedHigh;
    }
  }

  // -----------------------------------------------------------------------
  // Setup
  // -----------------------------------------------------------------------

  @BeforeAll
  static void setUp() throws IOException {
    sanityBuffer = createSanityBuffer(SANITY_BUFFER_SIZE);

    ObjectMapper mapper = new ObjectMapper();
    try (InputStream is =
        XXH3SanityTest.class.getResourceAsStream("/xxhash_test_vectors.json")) {
      if (is == null) {
        throw new IOException("xxhash_test_vectors.json not found on test classpath");
      }
      JsonNode root = mapper.readTree(is);

      xxh3_64BitsVectors = new ArrayList<>();
      for (JsonNode v : root.get("xxh3_64bits")) {
        xxh3_64BitsVectors.add(new TestData64(
            v.get("len").asInt(),
            v.get("seed").asLong(),
            v.get("expected").asLong()));
      }

      xxh3_128BitsVectors = new ArrayList<>();
      for (JsonNode v : root.get("xxh3_128bits")) {
        xxh3_128BitsVectors.add(new TestData128(
            v.get("len").asInt(),
            v.get("seed").asLong(),
            v.get("expectedLow").asLong(),
            v.get("expectedHigh").asLong()));
      }
    }
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  /** Pseudorandom buffer matching C fillTestBuffer. */
  public static byte[] createSanityBuffer(int length) {
    byte[] buf = new byte[length];
    long gen = PRIME32;
    for (int i = 0; i < length; i++) {
      buf[i] = (byte) (gen >>> 56);
      gen *= PRIME64;
    }
    return buf;
  }

  private static byte[] inputOf(int len) {
    if (len == 0) return new byte[0];
    byte[] out = new byte[len];
    System.arraycopy(sanityBuffer, 0, out, 0, len);
    return out;
  }

  /** Decodes the byte[16] big-endian [high64 || low64] result from hashUnsafeBytes128. */
  private static long[] decode128(byte[] result) {
    ByteBuffer bb = ByteBuffer.wrap(result).order(ByteOrder.BIG_ENDIAN);
    long high64 = bb.getLong(0);
    long low64  = bb.getLong(8);
    return new long[]{low64, high64};
  }

  // -----------------------------------------------------------------------
  // 64-bit: seeded vectors from JSON
  // -----------------------------------------------------------------------

  @ParameterizedTest
  @MethodSource("xxh3_64BitsTestData")
  @DisplayName("XXH3 64-bit: JSON test vectors (seeded)")
  void testXXH3_64bits_seeded(TestData64 td) {
    byte[] input = inputOf(td.len);

    long result = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, td.len, td.seed);
    assertEquals(td.expected, result,
        String.format("XXH3_64bits_withSeed mismatch for len=%d seed=0x%016X", td.len, td.seed));

    if (td.seed == 0L) {
      long resultNoSeed = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, td.len);
      assertEquals(td.expected, resultNoSeed,
          String.format("XXH3_64bits (seed=0) mismatch for len=%d", td.len));
    }
  }

  static Stream<TestData64> xxh3_64BitsTestData() {
    return xxh3_64BitsVectors.stream();
  }

  // -----------------------------------------------------------------------
  // 128-bit: all seeded vectors from JSON
  // -----------------------------------------------------------------------

  @ParameterizedTest
  @MethodSource("xxh3_128BitsTestData")
  @DisplayName("XXH3 128-bit: JSON test vectors (seeded)")
  void testXXH3_128bits_seeded(TestData128 td) {
    byte[] input = inputOf(td.len);

    byte[] raw = XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, td.len, td.seed);
    long[] decoded = decode128(raw);
    assertEquals(td.expectedLow, decoded[0],
        String.format("XXH3_128bits_withSeed low64 mismatch for len=%d seed=0x%016X",
            td.len, td.seed));
    assertEquals(td.expectedHigh, decoded[1],
        String.format("XXH3_128bits_withSeed high64 mismatch for len=%d seed=0x%016X",
            td.len, td.seed));

    if (td.seed == 0L) {
      byte[] rawNoSeed = XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, td.len);
      assertArrayEquals(raw, rawNoSeed,
          String.format("XXH3_128bits (seed=0) must match no-seed for len=%d", td.len));
    }
  }

  static Stream<TestData128> xxh3_128BitsTestData() {
    return xxh3_128BitsVectors.stream();
  }

  // -----------------------------------------------------------------------
  // Known-value spot checks (from HashExpressionsSuite.scala)
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("XXH3 64-bit known values (empty and single-byte)")
  void testKnownValues64() {
    assertEquals(3244421341483603138L,
        XXH3.hashUnsafeBytes64(new byte[0], Platform.BYTE_ARRAY_OFFSET, 0),
        "XXH3_64bits([]) should match reference");
    assertEquals(-4302098779834749733L,
        XXH3.hashUnsafeBytes64(new byte[]{0}, Platform.BYTE_ARRAY_OFFSET, 1),
        "XXH3_64bits([0]) should match reference");
  }

  @Test
  @DisplayName("XXH3 128-bit known values (empty and single-byte)")
  void testKnownValues128() {
    byte[] emptyResult = XXH3.hashUnsafeBytes128(new byte[0], Platform.BYTE_ARRAY_OFFSET, 0);
    long[] emptyDecoded = decode128(emptyResult);
    assertEquals(6918025063187695999L,  emptyDecoded[0], "XXH3_128bits([]) low64");
    assertEquals(-7374073936536430376L, emptyDecoded[1], "XXH3_128bits([]) high64");

    byte[] singleResult = XXH3.hashUnsafeBytes128(new byte[]{0}, Platform.BYTE_ARRAY_OFFSET, 1);
    long[] singleDecoded = decode128(singleResult);
    assertEquals(-4302098779834749733L, singleDecoded[0], "XXH3_128bits([0]) low64");
    assertEquals(-6427377105285148822L, singleDecoded[1], "XXH3_128bits([0]) high64");
  }

  // -----------------------------------------------------------------------
  // hashInt / hashLong consistency
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("hashInt / hashLong must match hashUnsafeBytes on the same bytes")
  void testIntLongConsistency() {
    int  intVal  = 0x12345678;
    long longVal = 0x1234567890ABCDEFL;

    // hashInt(x) must equal hashUnsafeBytes on its 4-byte LE representation
    byte[] intBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(intVal).array();
    assertEquals(
        XXH3.hashInt64(intVal),
        XXH3.hashUnsafeBytes64(intBytes, Platform.BYTE_ARRAY_OFFSET, 4),
        "hashInt must equal hashUnsafeBytes on LE bytes");

    // hashLong(x) must equal hashUnsafeBytes on its 8-byte LE representation
    byte[] longBytes =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(longVal).array();
    assertEquals(
        XXH3.hashLong64(longVal),
        XXH3.hashUnsafeBytes64(longBytes, Platform.BYTE_ARRAY_OFFSET, 8),
        "hashLong must equal hashUnsafeBytes on LE bytes");
  }

  @Test
  @DisplayName("Seeded hashInt / hashLong must match seeded hashUnsafeBytes")
  void testSeededIntLongConsistency() {
    int  intVal  = 0xDEADBEEF;
    long longVal = 0xCAFEBABEDEADBEEFL;
    long seed    = 0x123456789ABCDEFL;

    byte[] intBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(intVal).array();
    assertEquals(
        XXH3.hashInt64(intVal, seed),
        XXH3.hashUnsafeBytes64(intBytes, Platform.BYTE_ARRAY_OFFSET, 4, seed),
        "seeded hashInt must equal seeded hashUnsafeBytes on LE bytes");

    byte[] longBytes =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(longVal).array();
    assertEquals(
        XXH3.hashLong64(longVal, seed),
        XXH3.hashUnsafeBytes64(longBytes, Platform.BYTE_ARRAY_OFFSET, 8, seed),
        "seeded hashLong must equal seeded hashUnsafeBytes on LE bytes");
  }

  @Test
  @DisplayName("Seeded hashInt128 / hashLong128 must match seeded hashUnsafeBytes128")
  void testSeeded128IntLongConsistency() {
    int  intVal  = 0xDEADBEEF;
    long longVal = 0xCAFEBABEDEADBEEFL;
    long seed    = 0x123456789ABCDEFL;

    byte[] intBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(intVal).array();
    assertArrayEquals(
        XXH3.hashInt128(intVal, seed),
        XXH3.hashUnsafeBytes128(intBytes, Platform.BYTE_ARRAY_OFFSET, 4, seed),
        "seeded hashInt128 must equal seeded hashUnsafeBytes128 on LE bytes");

    byte[] longBytes =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(longVal).array();
    assertArrayEquals(
        XXH3.hashLong128(longVal, seed),
        XXH3.hashUnsafeBytes128(longBytes, Platform.BYTE_ARRAY_OFFSET, 8, seed),
        "seeded hashLong128 must equal seeded hashUnsafeBytes128 on LE bytes");
  }

  // -----------------------------------------------------------------------
  // Seed=0 vs no-seed API must agree
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("No-seed API must agree with seed=0 for all sizes (64-bit and 128-bit)")
  void testNoSeedMatchesSeedZero() {
    int[] sizes = {0, 1, 4, 8, 16, 17, 32, 64, 128, 240, 241, 512, 1024};
    for (int sz : sizes) {
      byte[] input = inputOf(sz);
      assertEquals(
          XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz),
          XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz, 0L),
          "64-bit no-seed and seed=0 must match for len=" + sz);
      assertArrayEquals(
          XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz),
          XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz, 0L),
          "128-bit no-seed and seed=0 must match for len=" + sz);
    }
  }

  // -----------------------------------------------------------------------
  // Different seeds must produce different results
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("Different seeds must produce different 64-bit hashes for non-empty input")
  void testSeedSensitivity64() {
    int[] sizes = {1, 4, 8, 16, 17, 32, 64, 128, 240, 241, 512};
    for (int sz : sizes) {
      byte[] input = inputOf(sz);
      long h0 = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz, 0L);
      long h1 = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz, 1L);
      assertNotEquals(h0, h1,
          "seed=0 and seed=1 must differ for len=" + sz);
    }
  }

  @Test
  @DisplayName("Different seeds must produce different 128-bit hashes for non-empty input")
  void testSeedSensitivity128() {
    int[] sizes = {1, 4, 8, 16, 17, 32, 64, 128, 240, 241, 512};
    for (int sz : sizes) {
      byte[] input = inputOf(sz);
      long[] h0 = decode128(XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz, 0L));
      long[] h1 = decode128(XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz, 1L));
      assertTrue(h0[0] != h1[0] || h0[1] != h1[1],
          "seed=0 and seed=1 128-bit results must differ for len=" + sz);
    }
  }

  // -----------------------------------------------------------------------
  // Algorithm boundary sizes
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("Critical size boundaries must not throw")
  void testBoundaryConditions() {
    int[] criticalSizes = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 239, 240, 241,
        255, 256, 257
    };
    for (int sz : criticalSizes) {
      byte[] input = inputOf(sz);
      assertDoesNotThrow(() -> {
        XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz);
        XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz, PRIME32);
        XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz);
      }, "boundary condition must not throw for len=" + sz);
    }
  }

  // -----------------------------------------------------------------------
  // Large inputs
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("Large inputs must not throw and must produce non-zero hashes")
  void testLargeInputs() {
    int[] sizes = {1024, 2048, 4096, 8192, 16384};
    for (int sz : sizes) {
      byte[] input = createSanityBuffer(sz);
      long h = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz);
      assertNotEquals(0L, h, "large input 64-bit hash should not be zero for len=" + sz);

      byte[] h128 = XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz);
      long[] d = decode128(h128);
      assertTrue(d[0] != 0L || d[1] != 0L,
          "large input 128-bit hash should not be all zeros for len=" + sz);
    }
  }

  // -----------------------------------------------------------------------
  // Determinism
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("Hash results must be deterministic")
  void testDeterminism() {
    int[] sizes = {0, 1, 16, 17, 128, 241, 1024};
    for (int sz : sizes) {
      byte[] input = inputOf(sz);
      long h1 = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz);
      long h2 = XXH3.hashUnsafeBytes64(input, Platform.BYTE_ARRAY_OFFSET, sz);
      assertEquals(h1, h2, "64-bit hash must be deterministic for len=" + sz);

      byte[] r1 = XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz);
      byte[] r2 = XXH3.hashUnsafeBytes128(input, Platform.BYTE_ARRAY_OFFSET, sz);
      assertArrayEquals(r1, r2, "128-bit hash must be deterministic for len=" + sz);
    }
  }

  // (Removed: testVectorScalarAgreement. The Vector API path was deleted from
  // XXH3.java because it was consistently slower than the scalar Opt-1 path.)

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------

  @Test
  @DisplayName("Test vector summary")
  void testSummaryStatistics() {
    int count64  = xxh3_64BitsVectors.size();
    int count128 = xxh3_128BitsVectors.size();
    System.out.printf("XXH3 sanity tests - 64-bit vectors: %d, 128-bit vectors: %d%n",
        count64, count128);
    assertTrue(count64  > 0, "Must have 64-bit test vectors");
    assertTrue(count128 > 0, "Must have 128-bit test vectors");
  }
}
