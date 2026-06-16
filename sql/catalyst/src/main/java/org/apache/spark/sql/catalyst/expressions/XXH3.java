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
 * xxHash - Extremely Fast Hash algorithm
 * Based on the original C implementation by Yann Collet
 * Copyright (C) 2012-2023 Yann Collet
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * The long-input scalar path uses the "Opt-1" transformation: the 8
 * accumulators live as method-local longs (a0..a7) across the entire
 * stripe + scramble + merge sequence, so HotSpot C2 keeps them in CPU
 * registers throughout. This avoids the heap load/store traffic that the
 * traditional long[] accumulator pattern produces when escape analysis
 * fails across the original 4-level call chain.
 *
 * Memory access goes through Platform.getLong (sun.misc.Unsafe under the
 * hood) with the receiver typed as Object - matching this class's public
 * API (which accepts both heap byte[] and Spark off-heap memory).
 *
 * Reference C implementation lines refer to xxhash.h from the xxHash project.
 */
package org.apache.spark.sql.catalyst.expressions;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

import org.apache.spark.unsafe.Platform;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * XXH3 and XXH128 hash functions (Yann Collet) - high quality, fast 64-bit
 * and 128-bit hash codes.
 *
 * <p>Algorithm structure mirrors the C reference (xxhash.h):
 * <ul>
 *   <li>0-16 bytes:    {@code XXH3_len_0to16_64b}     (xxhash.h ~line 4729)
 *   <li>17-128 bytes:  {@code XXH3_len_17to128_64b}   (xxhash.h ~line 4799)
 *   <li>129-240 bytes: {@code XXH3_len_129to240_64b}
 *   <li>&gt;240 bytes: Opt-1 inlined accumulator hash (composes
 *                      {@code XXH3_hashLong_internal_loop} + per-stripe
 *                      {@code XXH3_accumulate_512_scalar} + per-block
 *                      {@code XXH3_scrambleAcc_scalar} + {@code XXH3_mergeAccs}
 *                      with the accumulators kept as locals a0..a7).
 * </ul>
 *
 * <p>The 64-bit API returns long: {@link #hashInt64}, {@link #hashLong64},
 * {@link #hashUnsafeBytes64}, {@link #hashUnsafeWords64}, {@link #hashUTF8String64}.
 *
 * <p>The 128-bit API returns byte[16] in canonical big-endian form
 * [high64 || low64]: {@link #hashInt128}, {@link #hashLong128},
 * {@link #hashUnsafeBytes128}, {@link #hashUTF8String128}.
 */
public final class XXH3 {

  private static final boolean IS_BIG_ENDIAN =
      ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN);

  /** Absolute base offset for Platform reads into a byte[]. */
  private static final long SEC_BASE = Platform.BYTE_ARRAY_OFFSET;

  // ==========================================================================
  // Constants (verbatim from xxhash.h)
  // ==========================================================================

  /** Pseudorandom secret taken directly from FARSH (from xxhash.h, XXH3_kSecret). */
  private static final byte[] XXH3_kSecret = {
    (byte)0xb8, (byte)0xfe, (byte)0x6c, (byte)0x39, (byte)0x23, (byte)0xa4, (byte)0x4b, (byte)0xbe,
    (byte)0x7c, (byte)0x01, (byte)0x81, (byte)0x2c, (byte)0xf7, (byte)0x21, (byte)0xad, (byte)0x1c,
    (byte)0xde, (byte)0xd4, (byte)0x6d, (byte)0xe9, (byte)0x83, (byte)0x90, (byte)0x97, (byte)0xdb,
    (byte)0x72, (byte)0x40, (byte)0xa4, (byte)0xa4, (byte)0xb7, (byte)0xb3, (byte)0x67, (byte)0x1f,
    (byte)0xcb, (byte)0x79, (byte)0xe6, (byte)0x4e, (byte)0xcc, (byte)0xc0, (byte)0xe5, (byte)0x78,
    (byte)0x82, (byte)0x5a, (byte)0xd0, (byte)0x7d, (byte)0xcc, (byte)0xff, (byte)0x72, (byte)0x21,
    (byte)0xb8, (byte)0x08, (byte)0x46, (byte)0x74, (byte)0xf7, (byte)0x43, (byte)0x24, (byte)0x8e,
    (byte)0xe0, (byte)0x35, (byte)0x90, (byte)0xe6, (byte)0x81, (byte)0x3a, (byte)0x26, (byte)0x4c,
    (byte)0x3c, (byte)0x28, (byte)0x52, (byte)0xbb, (byte)0x91, (byte)0xc3, (byte)0x00, (byte)0xcb,
    (byte)0x88, (byte)0xd0, (byte)0x65, (byte)0x8b, (byte)0x1b, (byte)0x53, (byte)0x2e, (byte)0xa3,
    (byte)0x71, (byte)0x64, (byte)0x48, (byte)0x97, (byte)0xa2, (byte)0x0d, (byte)0xf9, (byte)0x4e,
    (byte)0x38, (byte)0x19, (byte)0xef, (byte)0x46, (byte)0xa9, (byte)0xde, (byte)0xac, (byte)0xd8,
    (byte)0xa8, (byte)0xfa, (byte)0x76, (byte)0x3f, (byte)0xe3, (byte)0x9c, (byte)0x34, (byte)0x3f,
    (byte)0xf9, (byte)0xdc, (byte)0xbb, (byte)0xc7, (byte)0xc7, (byte)0x0b, (byte)0x4f, (byte)0x1d,
    (byte)0x8a, (byte)0x51, (byte)0xe0, (byte)0x4b, (byte)0xcd, (byte)0xb4, (byte)0x59, (byte)0x31,
    (byte)0xc8, (byte)0x9f, (byte)0x7e, (byte)0xc9, (byte)0xd9, (byte)0x78, (byte)0x73, (byte)0x64,
    (byte)0xea, (byte)0xc5, (byte)0xac, (byte)0x83, (byte)0x34, (byte)0xd3, (byte)0xeb, (byte)0xc3,
    (byte)0xc5, (byte)0x81, (byte)0xa0, (byte)0xff, (byte)0xfa, (byte)0x13, (byte)0x63, (byte)0xeb,
    (byte)0x17, (byte)0x0d, (byte)0xdd, (byte)0x51, (byte)0xb7, (byte)0xf0, (byte)0xda, (byte)0x49,
    (byte)0xd3, (byte)0x16, (byte)0x55, (byte)0x26, (byte)0x29, (byte)0xd4, (byte)0x68, (byte)0x9e,
    (byte)0x2b, (byte)0x16, (byte)0xbe, (byte)0x58, (byte)0x7d, (byte)0x47, (byte)0xa1, (byte)0xfc,
    (byte)0x8f, (byte)0xf8, (byte)0xb8, (byte)0xd1, (byte)0x7a, (byte)0xd0, (byte)0x31, (byte)0xce,
    (byte)0x45, (byte)0xcb, (byte)0x3a, (byte)0x8f, (byte)0x95, (byte)0x16, (byte)0x04, (byte)0x28,
    (byte)0xaf, (byte)0xd7, (byte)0xfb, (byte)0xca, (byte)0xbb, (byte)0x4b, (byte)0x40, (byte)0x7e,
  };

  // Precomputed bitflip constants for the inlined primitive hashers (hashInt64 /
  // hashLong64 / hashInt128 / hashLong128). JIT folds these into machine-code
  // constants and avoids re-deriving them every call.
  //   64-bit  path: kSecret[8..15]  XOR kSecret[16..23]
  //   128-bit path: kSecret[16..23] XOR kSecret[24..31]
  private static final long BITFLIP_64_4_8;
  private static final long BITFLIP_128_4_8;
  static {
    BITFLIP_64_4_8  = getLE64(XXH3_kSecret, SEC_BASE + 8)  ^ getLE64(XXH3_kSecret, SEC_BASE + 16);
    BITFLIP_128_4_8 = getLE64(XXH3_kSecret, SEC_BASE + 16) ^ getLE64(XXH3_kSecret, SEC_BASE + 24);
  }

  private static final int XXH3_SECRET_SIZE_MIN     = 136;
  private static final int XXH3_SECRET_DEFAULT_SIZE = 192;
  private static final int XXH_SECRET_MERGEACCS_START = 11;
  private static final int XXH_SECRET_LASTACC_START   = 7;
  private static final int XXH3_MIDSIZE_MAX         = 240;
  private static final int XXH3_MIDSIZE_STARTOFFSET = 3;
  private static final int XXH3_MIDSIZE_LASTOFFSET  = 17;

  private static final long XXH_PRIME32_1 = 0x9E3779B1L;
  private static final long XXH_PRIME32_2 = 0x85EBCA77L;
  private static final long XXH_PRIME32_3 = 0xC2B2AE3DL;

  private static final long XXH_PRIME64_1 = 0x9E3779B185EBCA87L;
  private static final long XXH_PRIME64_2 = 0xC2B2AE3D27D4EB4FL;
  private static final long XXH_PRIME64_3 = 0x165667B19E3779F9L;
  private static final long XXH_PRIME64_4 = 0x85EBCA77C2B2AE63L;
  private static final long XXH_PRIME64_5 = 0x27D4EB2F165667C5L;

  private static final long PRIME_MX1 = 0x165667919E3779F9L;
  private static final long PRIME_MX2 = 0x9FB21C651E98DF25L;

  private static final int XXH_STRIPE_LEN          = 64;
  private static final int XXH_SECRET_CONSUME_RATE = 8;

  // Default-secret-derived constants. With kSecret being 192 bytes, the inner
  // block has 16 stripes (= (192-64)/8) of 64 bytes each = 1024 bytes total.
  // Hardcoded so C2 sees compile-time-constant trip counts and offsets.
  private static final int  NB_STRIPES_PER_BLOCK = 16;
  private static final int  BLOCK_LEN            = XXH_STRIPE_LEN * NB_STRIPES_PER_BLOCK;
  private static final long SCRAM_OFFSET         = XXH3_SECRET_DEFAULT_SIZE - XXH_STRIPE_LEN;
  private static final long LAST_ACC_OFFSET      =
      XXH3_SECRET_DEFAULT_SIZE - XXH_STRIPE_LEN - XXH_SECRET_LASTACC_START;

  // ==========================================================================
  // Platform read/write helpers (Object-typed; receiver may be byte[] heap
  // or null + raw address for off-heap memory).
  // ==========================================================================

  private static long getLE64(Object base, long off) {
    long v = Platform.getLong(base, off);
    return IS_BIG_ENDIAN ? Long.reverseBytes(v) : v;
  }

  private static int getLE32(Object base, long off) {
    int v = Platform.getInt(base, off);
    return IS_BIG_ENDIAN ? Integer.reverseBytes(v) : v;
  }

  private static int getU8(Object base, long off) {
    return Platform.getByte(base, off) & 0xFF;
  }

  private static void putLE64(byte[] arr, long off, long val) {
    Platform.putLong(arr, off, IS_BIG_ENDIAN ? Long.reverseBytes(val) : val);
  }

  // byte[]-typed helper. Identical to getLE64 above but with a typed receiver
  // so C2 can apply array-specific alias analysis.
  private static long getLE64BA(byte[] arr, long off) {
    long v = Platform.getLong(arr, off);
    return IS_BIG_ENDIAN ? Long.reverseBytes(v) : v;
  }

  // ==========================================================================
  // VarHandle read/write helpers (byte[]-only; zero-based int index).
  //
  // VarHandle.byteArrayViewVarHandle already performs LE/BE conversion, so no
  // IS_BIG_ENDIAN check is needed here. Used exclusively in the hashBytes64/hashBytes128 paths.
  // ==========================================================================

  private static final VarHandle LE_LONG =
      MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
  private static final VarHandle LE_INT =
      MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

  private static long getLE64VH(byte[] arr, int idx) {
    return (long) LE_LONG.get(arr, idx);
  }

  private static int getLE32VH(byte[] arr, int idx) {
    return (int) LE_INT.get(arr, idx);
  }

  private static void putLE64VH(byte[] arr, int idx, long val) {
    LE_LONG.set(arr, idx, val);
  }

  // ==========================================================================
  // Mixing primitives - direct translations of xxhash.h.
  // ==========================================================================

  /** Mirrors XXH64_avalanche (xxhash.h). */
  private static long XXH64_avalanche(long h) {
    h ^= h >>> 33;
    h *= XXH_PRIME64_2;
    h ^= h >>> 29;
    h *= XXH_PRIME64_3;
    h ^= h >>> 32;
    return h;
  }

  /** Mirrors XXH3_avalanche (xxhash.h). */
  private static long XXH3_avalanche(long h) {
    h ^= h >>> 37;
    h *= PRIME_MX1;
    h ^= h >>> 32;
    return h;
  }

  /** Mirrors XXH3_rrmxmx (xxhash.h). */
  private static long XXH3_rrmxmx(long hash, long len) {
    hash ^= Long.rotateLeft(hash, 49) ^ Long.rotateLeft(hash, 24);
    hash *= PRIME_MX2;
    hash ^= (hash >>> 35) + len;
    hash *= PRIME_MX2;
    hash ^= hash >>> 28;
    return hash;
  }

  /** Low 64 bits of 64x64 -> 128 multiply. */
  private static long XXH_mult128_low64(long lhs, long rhs) {
    return lhs * rhs;
  }

  /** High 64 bits of UNSIGNED 64x64 multiply (Math.multiplyHigh is signed). */
  private static long XXH_mult128_high64(long lhs, long rhs) {
    long high = Math.multiplyHigh(lhs, rhs);
    if (lhs < 0) high += rhs;
    if (rhs < 0) high += lhs;
    return high;
  }

  /** Folded 128-bit multiply: {@code low64 ^ high64}. Mirrors XXH3_mul128_fold64. */
  private static long XXH3_mul128_fold64(long lhs, long rhs) {
    return XXH_mult128_low64(lhs, rhs) ^ XXH_mult128_high64(lhs, rhs);
  }

  // ==========================================================================
  // Small-input 64-bit helpers (Object-typed). Mirror XXH3_len_*_64b in
  // xxhash.h (~lines 4700-4900).
  // ==========================================================================

  /** Mirrors XXH3_len_1to3_64b. */
  private static long XXH3_len_1to3_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    int c1 = getU8(in, off);
    int c2 = getU8(in, off + (len >>> 1));
    int c3 = getU8(in, off + len - 1);
    int combined = (c1 << 16) | (c2 << 24) | c3 | (len << 8);
    long bitflip = ((getLE32(secret, secOff) ^ getLE32(secret, secOff + 4)) & 0xFFFFFFFFL) + seed;
    return XXH64_avalanche((combined & 0xFFFFFFFFL) ^ bitflip);
  }

  /** Mirrors XXH3_len_4to8_64b. */
  private static long XXH3_len_4to8_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
    long input1 = getLE32(in, off) & 0xFFFFFFFFL;
    long input2 = getLE32(in, off + len - 4) & 0xFFFFFFFFL;
    long bitflip = (getLE64(secret, secOff + 8) ^ getLE64(secret, secOff + 16)) - seed;
    long input64 = input2 + (input1 << 32);
    return XXH3_rrmxmx(input64 ^ bitflip, len);
  }

  /** Mirrors XXH3_len_9to16_64b. */
  private static long XXH3_len_9to16_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    long bitflip1 = (getLE64(secret, secOff + 24) ^ getLE64(secret, secOff + 32)) + seed;
    long bitflip2 = (getLE64(secret, secOff + 40) ^ getLE64(secret, secOff + 48)) - seed;
    long lo = getLE64(in, off)           ^ bitflip1;
    long hi = getLE64(in, off + len - 8) ^ bitflip2;
    long acc = len + Long.reverseBytes(lo) + hi + XXH3_mul128_fold64(lo, hi);
    return XXH3_avalanche(acc);
  }

  /** Mirrors XXH3_len_0to16_64b (dispatch by length). */
  private static long XXH3_len_0to16_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    if (len > 8)  return XXH3_len_9to16_64b(in, off, len, secret, secOff, seed);
    if (len >= 4) return XXH3_len_4to8_64b(in, off, len, secret, secOff, seed);
    if (len > 0)  return XXH3_len_1to3_64b(in, off, len, secret, secOff, seed);
    return XXH64_avalanche(seed ^ getLE64(secret, secOff + 56) ^ getLE64(secret, secOff + 64));
  }

  /** Mirrors XXH3_mix16B. */
  private static long XXH3_mix16B(
      Object in, long inOff, byte[] secret, long secOff, long seed) {
    return XXH3_mul128_fold64(
      getLE64(in, inOff)     ^ (getLE64(secret, secOff)     + seed),
      getLE64(in, inOff + 8) ^ (getLE64(secret, secOff + 8) - seed));
  }

  /** Mirrors XXH3_len_17to128_64b. */
  private static long XXH3_len_17to128_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    long acc = len * XXH_PRIME64_1;
    if (len > 32) {
      if (len > 64) {
        if (len > 96) {
          acc += XXH3_mix16B(in, off + 48,       secret, secOff + 96,  seed);
          acc += XXH3_mix16B(in, off + len - 64, secret, secOff + 112, seed);
        }
        acc += XXH3_mix16B(in, off + 32,       secret, secOff + 64, seed);
        acc += XXH3_mix16B(in, off + len - 48, secret, secOff + 80, seed);
      }
      acc += XXH3_mix16B(in, off + 16,       secret, secOff + 32, seed);
      acc += XXH3_mix16B(in, off + len - 32, secret, secOff + 48, seed);
    }
    acc += XXH3_mix16B(in, off,            secret, secOff,      seed);
    acc += XXH3_mix16B(in, off + len - 16, secret, secOff + 16, seed);
    return XXH3_avalanche(acc);
  }

  /** Mirrors XXH3_len_129to240_64b. */
  private static long XXH3_len_129to240_64b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    long acc = len * XXH_PRIME64_1;
    int nbRounds = len / 16;
    for (int i = 0; i < 8; i++) {
      acc += XXH3_mix16B(in, off + 16 * i, secret, secOff + 16 * i, seed);
    }
    long accEnd = XXH3_mix16B(
      in, off + len - 16,
      secret, secOff + XXH3_SECRET_SIZE_MIN - XXH3_MIDSIZE_LASTOFFSET, seed);
    acc = XXH3_avalanche(acc);
    for (int i = 8; i < nbRounds; i++) {
      accEnd += XXH3_mix16B(
        in, off + 16 * i,
        secret, secOff + 16 * (i - 8) + XXH3_MIDSIZE_STARTOFFSET, seed);
    }
    return XXH3_avalanche(acc + accEnd);
  }

  // ==========================================================================
  // Small-input 128-bit helpers (Object-typed).
  // ==========================================================================

  private static long XXH128_mix32B_once(
      long seed, byte[] secret, long secOff, long acc,
      long i0, long i1, long i2, long i3) {
    acc += XXH3_mul128_fold64(
      i0 ^ (getLE64(secret, secOff)     + seed),
      i1 ^ (getLE64(secret, secOff + 8) - seed));
    return acc ^ (i2 + i3);
  }

  /** Mirrors XXH3_len_0to16_128b. */
  private static byte[] XXH3_len_0to16_128b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    if (len > 8) {
      long bitflipl = (getLE64(secret, secOff + 32) ^ getLE64(secret, secOff + 40)) - seed;
      long bitfliph = (getLE64(secret, secOff + 48) ^ getLE64(secret, secOff + 56)) + seed;
      long lo = getLE64(in, off);
      long hi = getLE64(in, off + len - 8);
      long loHigh = XXH_mult128_low64(lo ^ hi ^ bitflipl, XXH_PRIME64_1);
      long hiHigh = XXH_mult128_high64(lo ^ hi ^ bitflipl, XXH_PRIME64_1);
      loHigh += ((long)(len - 1) << 54);
      hi ^= bitfliph;
      hiHigh += hi + (hi & 0xFFFFFFFFL) * (XXH_PRIME32_2 - 1);
      loHigh ^= Long.reverseBytes(hiHigh);
      long h128lo = XXH_mult128_low64(loHigh, XXH_PRIME64_2);
      long h128hi = XXH_mult128_high64(loHigh, XXH_PRIME64_2) + hiHigh * XXH_PRIME64_2;
      return toByteArray128(XXH3_avalanche(h128hi), XXH3_avalanche(h128lo));
    }
    if (len >= 4) {
      seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
      int inLo = getLE32(in, off);
      int inHi = getLE32(in, off + len - 4);
      long in64 = (inLo & 0xFFFFFFFFL) + (((long)(inHi & 0xFFFFFFFFL)) << 32);
      long bitflip = (getLE64(secret, secOff + 16) ^ getLE64(secret, secOff + 24)) + seed;
      long keyed = in64 ^ bitflip;
      long multiplier = XXH_PRIME64_1 + (len << 2);
      long mLow  = XXH_mult128_low64(keyed, multiplier);
      long mHigh = XXH_mult128_high64(keyed, multiplier);
      mHigh += mLow << 1;
      mLow  ^= mHigh >>> 3;
      long low  = mLow ^ (mLow >>> 35);
      low  *= PRIME_MX2;
      low  ^= low >>> 28;
      return toByteArray128(XXH3_avalanche(mHigh), low);
    }
    if (len > 0) {
      int c1 = getU8(in, off);
      int c2 = getU8(in, off + (len >> 1));
      int c3 = getU8(in, off + len - 1);
      int combinedl = (c1 << 16) | (c2 << 24) | c3 | (len << 8);
      int combinedh = Integer.rotateLeft(Integer.reverseBytes(combinedl), 13);
      long bitflipl =
          ((long)(getLE32(secret, secOff) ^ getLE32(secret, secOff + 4)) & 0xFFFFFFFFL) + seed;
      long bitfliph =
          ((long)(getLE32(secret, secOff + 8) ^ getLE32(secret, secOff + 12)) & 0xFFFFFFFFL) - seed;
      return toByteArray128(
        XXH64_avalanche((combinedh & 0xFFFFFFFFL) ^ bitfliph),
        XXH64_avalanche((combinedl & 0xFFFFFFFFL) ^ bitflipl));
    }
    long bitflipl = getLE64(secret, secOff + 64) ^ getLE64(secret, secOff + 72);
    long bitfliph = getLE64(secret, secOff + 80) ^ getLE64(secret, secOff + 88);
    return toByteArray128(
      XXH64_avalanche(seed ^ bitfliph),
      XXH64_avalanche(seed ^ bitflipl));
  }

  /** Mirrors XXH3_len_17to128_128b. */
  private static byte[] XXH3_len_17to128_128b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    long acc0 = len * XXH_PRIME64_1;
    long acc1 = 0;
    if (len > 32) {
      if (len > 64) {
        if (len > 96) {
          long i0 = getLE64(in, off + 48),       i1 = getLE64(in, off + 56);
          long i2 = getLE64(in, off + len - 64), i3 = getLE64(in, off + len - 56);
          acc0 = XXH128_mix32B_once(seed, secret, secOff + 96,      acc0, i0, i1, i2, i3);
          acc1 = XXH128_mix32B_once(seed, secret, secOff + 96 + 16, acc1, i2, i3, i0, i1);
        }
        long i0 = getLE64(in, off + 32),       i1 = getLE64(in, off + 40);
        long i2 = getLE64(in, off + len - 48), i3 = getLE64(in, off + len - 40);
        acc0 = XXH128_mix32B_once(seed, secret, secOff + 64,      acc0, i0, i1, i2, i3);
        acc1 = XXH128_mix32B_once(seed, secret, secOff + 64 + 16, acc1, i2, i3, i0, i1);
      }
      long i0 = getLE64(in, off + 16),       i1 = getLE64(in, off + 24);
      long i2 = getLE64(in, off + len - 32), i3 = getLE64(in, off + len - 24);
      acc0 = XXH128_mix32B_once(seed, secret, secOff + 32,      acc0, i0, i1, i2, i3);
      acc1 = XXH128_mix32B_once(seed, secret, secOff + 32 + 16, acc1, i2, i3, i0, i1);
    }
    long i0 = getLE64(in, off),            i1 = getLE64(in, off + 8);
    long i2 = getLE64(in, off + len - 16), i3 = getLE64(in, off + len - 8);
    acc0 = XXH128_mix32B_once(seed, secret, secOff,      acc0, i0, i1, i2, i3);
    acc1 = XXH128_mix32B_once(seed, secret, secOff + 16, acc1, i2, i3, i0, i1);
    long low  = XXH3_avalanche(acc0 + acc1);
    long high = 0L - XXH3_avalanche(acc0 * XXH_PRIME64_1 + acc1 * XXH_PRIME64_4
                                   + (len - seed) * XXH_PRIME64_2);
    return toByteArray128(high, low);
  }

  /** Mirrors XXH3_len_129to240_128b. */
  private static byte[] XXH3_len_129to240_128b(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    int nbRounds = len / 32;
    long acc0 = len * XXH_PRIME64_1;
    long acc1 = 0;
    int i = 0;
    for (; i < 4; ++i) {
      long i0 = getLE64(in, off + 32L * i),       i1 = getLE64(in, off + 32L * i + 8);
      long i2 = getLE64(in, off + 32L * i + 16),  i3 = getLE64(in, off + 32L * i + 24);
      acc0 = XXH128_mix32B_once(seed, secret, secOff + 32L * i,      acc0, i0, i1, i2, i3);
      acc1 = XXH128_mix32B_once(seed, secret, secOff + 32L * i + 16, acc1, i2, i3, i0, i1);
    }
    acc0 = XXH3_avalanche(acc0);
    acc1 = XXH3_avalanche(acc1);
    for (; i < nbRounds; ++i) {
      long i0 = getLE64(in, off + 32L * i),       i1 = getLE64(in, off + 32L * i + 8);
      long i2 = getLE64(in, off + 32L * i + 16),  i3 = getLE64(in, off + 32L * i + 24);
      acc0 = XXH128_mix32B_once(seed, secret,
        secOff + XXH3_MIDSIZE_STARTOFFSET + 32L * (i - 4),      acc0, i0, i1, i2, i3);
      acc1 = XXH128_mix32B_once(seed, secret,
        secOff + XXH3_MIDSIZE_STARTOFFSET + 32L * (i - 4) + 16, acc1, i2, i3, i0, i1);
    }
    long i0 = getLE64(in, off + len - 16), i1 = getLE64(in, off + len - 8);
    long i2 = getLE64(in, off + len - 32), i3 = getLE64(in, off + len - 24);
    acc0 = XXH128_mix32B_once(-seed, secret,
      secOff + XXH3_SECRET_SIZE_MIN - XXH3_MIDSIZE_LASTOFFSET - 16, acc0, i0, i1, i2, i3);
    acc1 = XXH128_mix32B_once(-seed, secret,
      secOff + XXH3_SECRET_SIZE_MIN - XXH3_MIDSIZE_LASTOFFSET,      acc1, i2, i3, i0, i1);
    long low  = XXH3_avalanche(acc0 + acc1);
    long high = 0L - XXH3_avalanche(acc0 * XXH_PRIME64_1 + acc1 * XXH_PRIME64_4
                                   + (len - seed) * XXH_PRIME64_2);
    return toByteArray128(high, low);
  }

  // ==========================================================================
  // Opt-1 long-input scalar hash (>240 bytes), Object-typed. The canonical
  // path. Replaces the C composition of
  //   XXH3_hashLong_internal_loop  (xxhash.h ~line 6231)
  //   + XXH3_accumulate_512_scalar (xxhash.h ~line 6032)
  //   + XXH3_scrambleAcc_scalar    (xxhash.h ~line 6081)
  //   + XXH3_mergeAccs
  // with a single monolithic method that keeps the 8 accumulators as locals
  // a0..a7 throughout. This lets C2 hold them in registers and avoids the
  // long[] heap traffic that plagued earlier ports.
  // ==========================================================================

  private static long XXH3_hashLong_64b(
      Object in, long off, int len, byte[] secret, long secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    long scramOff = secOff + SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      long blockBase = off + (long) n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        long b  = blockBase + (long) s * XXH_STRIPE_LEN;
        long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
        // XXH3_accumulate_512_scalar (xxhash.h:6032). Per-lane:
        //   xacc[lane ^ 1] += data_val;                            // lane-swap add
        //   xacc[lane] += mul32x32(data_key_lo, data_key_hi);      // mix-multiply
        // where data_key = data_val ^ secret_val. Spelled out for 8 lanes:
        long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
        long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
        long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
        long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
        long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
        long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
        long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
        long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      // XXH3_scrambleAcc_scalar (xxhash.h:6081). Per-lane:
      //   acc ^= acc >>> 47; acc ^= secret_key; acc *= XXH_PRIME32_1;
      a0 ^= a0 >>> 47; a0 ^= getLE64(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    // Partial stripes after the last full block (no scramble after partial).
    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    long partBase = off + (long) nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      long b  = partBase + (long) s * XXH_STRIPE_LEN;
      long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    // Last stripe (xxhash.h: "if (len > XXH_STRIPE_LEN)" branch).
    if (len > XXH_STRIPE_LEN) {
      long b  = off + len - XXH_STRIPE_LEN;
      long sk = secOff + LAST_ACC_OFFSET;
      long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    // Inlined XXH3_mergeAccs: 4 mul128_fold64 pairs + avalanche.
    long ms = secOff + XXH_SECRET_MERGEACCS_START;
    return XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms),      a1 ^ getLE64(secret, ms + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms + 16), a3 ^ getLE64(secret, ms + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms + 32), a5 ^ getLE64(secret, ms + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms + 48), a7 ^ getLE64(secret, ms + 56)));
  }

  /** 128-bit twin of XXH3_hashLong_64b. Same stripe loop, two merge calls. */
  private static byte[] XXH3_hashLong_128b(
      Object in, long off, int len, byte[] secret, long secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    long scramOff = secOff + SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      long blockBase = off + (long) n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        long b  = blockBase + (long) s * XXH_STRIPE_LEN;
        long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
        long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
        long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
        long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
        long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
        long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
        long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
        long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
        long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      a0 ^= a0 >>> 47; a0 ^= getLE64(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    long partBase = off + (long) nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      long b  = partBase + (long) s * XXH_STRIPE_LEN;
      long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    if (len > XXH_STRIPE_LEN) {
      long b  = off + len - XXH_STRIPE_LEN;
      long sk = secOff + LAST_ACC_OFFSET;
      long d0 = getLE64(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    long ms1 = secOff + XXH_SECRET_MERGEACCS_START;
    long low = XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms1),      a1 ^ getLE64(secret, ms1 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms1 + 16), a3 ^ getLE64(secret, ms1 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms1 + 32), a5 ^ getLE64(secret, ms1 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms1 + 48), a7 ^ getLE64(secret, ms1 + 56)));
    long ms2 = secOff + XXH3_SECRET_DEFAULT_SIZE - XXH_STRIPE_LEN - XXH_SECRET_MERGEACCS_START;
    long high = XXH3_avalanche(~(len * XXH_PRIME64_2)
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms2),      a1 ^ getLE64(secret, ms2 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms2 + 16), a3 ^ getLE64(secret, ms2 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms2 + 32), a5 ^ getLE64(secret, ms2 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms2 + 48), a7 ^ getLE64(secret, ms2 + 56)));
    return toByteArray128(high, low);
  }

  // ==========================================================================
  // byte[]-typed long-input hash. Identical algorithm to the Object-typed
  // version above; differs only in the receiver type of the input parameter
  // and the helper used to read from it. The byte[]-static input lets the
  // JIT apply array-specific alias analysis through the hot loop.
  //
  // (Secret reads still use getLE64 with byte[] secret, same as Object path.)
  // ==========================================================================

  private static long XXH3_hashLong_64b_ba(
      byte[] in, long off, int len, byte[] secret, long secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    long scramOff = secOff + SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      long blockBase = off + (long) n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        long b  = blockBase + (long) s * XXH_STRIPE_LEN;
        long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
        long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
        long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
        long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
        long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
        long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
        long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
        long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
        long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      a0 ^= a0 >>> 47; a0 ^= getLE64(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    long partBase = off + (long) nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      long b  = partBase + (long) s * XXH_STRIPE_LEN;
      long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    if (len > XXH_STRIPE_LEN) {
      long b  = off + len - XXH_STRIPE_LEN;
      long sk = secOff + LAST_ACC_OFFSET;
      long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    long ms = secOff + XXH_SECRET_MERGEACCS_START;
    return XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms),      a1 ^ getLE64(secret, ms + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms + 16), a3 ^ getLE64(secret, ms + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms + 32), a5 ^ getLE64(secret, ms + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms + 48), a7 ^ getLE64(secret, ms + 56)));
  }

  /** byte[]-typed twin of XXH3_hashLong_128b. */
  private static byte[] XXH3_hashLong_128b_ba(
      byte[] in, long off, int len, byte[] secret, long secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    long scramOff = secOff + SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      long blockBase = off + (long) n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        long b  = blockBase + (long) s * XXH_STRIPE_LEN;
        long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
        long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
        long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
        long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
        long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
        long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
        long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
        long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
        long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      a0 ^= a0 >>> 47; a0 ^= getLE64(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    long partBase = off + (long) nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      long b  = partBase + (long) s * XXH_STRIPE_LEN;
      long sk = secOff   + (long) s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    if (len > XXH_STRIPE_LEN) {
      long b  = off + len - XXH_STRIPE_LEN;
      long sk = secOff + LAST_ACC_OFFSET;
      long d0 = getLE64BA(in, b),      k0 = d0 ^ getLE64(secret, sk);
      long d1 = getLE64BA(in, b + 8),  k1 = d1 ^ getLE64(secret, sk + 8);
      long d2 = getLE64BA(in, b + 16), k2 = d2 ^ getLE64(secret, sk + 16);
      long d3 = getLE64BA(in, b + 24), k3 = d3 ^ getLE64(secret, sk + 24);
      long d4 = getLE64BA(in, b + 32), k4 = d4 ^ getLE64(secret, sk + 32);
      long d5 = getLE64BA(in, b + 40), k5 = d5 ^ getLE64(secret, sk + 40);
      long d6 = getLE64BA(in, b + 48), k6 = d6 ^ getLE64(secret, sk + 48);
      long d7 = getLE64BA(in, b + 56), k7 = d7 ^ getLE64(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    long ms1 = secOff + XXH_SECRET_MERGEACCS_START;
    long low = XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms1),      a1 ^ getLE64(secret, ms1 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms1 + 16), a3 ^ getLE64(secret, ms1 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms1 + 32), a5 ^ getLE64(secret, ms1 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms1 + 48), a7 ^ getLE64(secret, ms1 + 56)));
    long ms2 = secOff + XXH3_SECRET_DEFAULT_SIZE - XXH_STRIPE_LEN - XXH_SECRET_MERGEACCS_START;
    long high = XXH3_avalanche(~(len * XXH_PRIME64_2)
      + XXH3_mul128_fold64(a0 ^ getLE64(secret, ms2),      a1 ^ getLE64(secret, ms2 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64(secret, ms2 + 16), a3 ^ getLE64(secret, ms2 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64(secret, ms2 + 32), a5 ^ getLE64(secret, ms2 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64(secret, ms2 + 48), a7 ^ getLE64(secret, ms2 + 56)));
    return toByteArray128(high, low);
  }

  // ==========================================================================
  // VarHandle variants of the long-input hot loops (int offsets, no Unsafe).
  // Structure is identical to *_ba above; only the read helpers differ.
  // ==========================================================================

  private static long XXH3_hashLong_64b_vh(
      byte[] in, int off, int len, byte[] secret, int secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    int scramOff = secOff + (int) SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      int blockBase = off + n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        int b  = blockBase + s * XXH_STRIPE_LEN;
        int sk = secOff    + s * XXH_SECRET_CONSUME_RATE;
        long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
        long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
        long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
        long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
        long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
        long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
        long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
        long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      a0 ^= a0 >>> 47; a0 ^= getLE64VH(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64VH(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64VH(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64VH(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64VH(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64VH(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64VH(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64VH(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    int partBase  = off + nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      int b  = partBase + s * XXH_STRIPE_LEN;
      int sk = secOff   + s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
      long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
      long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
      long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
      long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
      long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
      long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
      long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    if (len > XXH_STRIPE_LEN) {
      int b  = off + len - XXH_STRIPE_LEN;
      int sk = secOff + (int) LAST_ACC_OFFSET;
      long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
      long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
      long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
      long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
      long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
      long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
      long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
      long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    int ms = secOff + XXH_SECRET_MERGEACCS_START;
    return XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64VH(secret, ms),      a1 ^ getLE64VH(secret, ms + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64VH(secret, ms + 16), a3 ^ getLE64VH(secret, ms + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64VH(secret, ms + 32), a5 ^ getLE64VH(secret, ms + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64VH(secret, ms + 48), a7 ^ getLE64VH(secret, ms + 56)));
  }

  private static byte[] XXH3_hashLong_128b_vh(
      byte[] in, int off, int len, byte[] secret, int secOff) {
    long a0 = XXH_PRIME32_3, a1 = XXH_PRIME64_1, a2 = XXH_PRIME64_2, a3 = XXH_PRIME64_3;
    long a4 = XXH_PRIME64_4, a5 = XXH_PRIME32_2, a6 = XXH_PRIME64_5, a7 = XXH_PRIME32_1;

    int nbBlocks = (len - 1) / BLOCK_LEN;
    int scramOff = secOff + (int) SCRAM_OFFSET;

    for (int n = 0; n < nbBlocks; n++) {
      int blockBase = off + n * BLOCK_LEN;
      for (int s = 0; s < NB_STRIPES_PER_BLOCK; s++) {
        int b  = blockBase + s * XXH_STRIPE_LEN;
        int sk = secOff    + s * XXH_SECRET_CONSUME_RATE;
        long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
        long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
        long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
        long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
        long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
        long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
        long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
        long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
        a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
        a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
        a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
        a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
        a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
        a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
        a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
        a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
      }
      a0 ^= a0 >>> 47; a0 ^= getLE64VH(secret, scramOff);      a0 *= XXH_PRIME32_1;
      a1 ^= a1 >>> 47; a1 ^= getLE64VH(secret, scramOff + 8);  a1 *= XXH_PRIME32_1;
      a2 ^= a2 >>> 47; a2 ^= getLE64VH(secret, scramOff + 16); a2 *= XXH_PRIME32_1;
      a3 ^= a3 >>> 47; a3 ^= getLE64VH(secret, scramOff + 24); a3 *= XXH_PRIME32_1;
      a4 ^= a4 >>> 47; a4 ^= getLE64VH(secret, scramOff + 32); a4 *= XXH_PRIME32_1;
      a5 ^= a5 >>> 47; a5 ^= getLE64VH(secret, scramOff + 40); a5 *= XXH_PRIME32_1;
      a6 ^= a6 >>> 47; a6 ^= getLE64VH(secret, scramOff + 48); a6 *= XXH_PRIME32_1;
      a7 ^= a7 >>> 47; a7 ^= getLE64VH(secret, scramOff + 56); a7 *= XXH_PRIME32_1;
    }

    int nbStripes = ((len - 1) - BLOCK_LEN * nbBlocks) / XXH_STRIPE_LEN;
    int partBase  = off + nbBlocks * BLOCK_LEN;
    for (int s = 0; s < nbStripes; s++) {
      int b  = partBase + s * XXH_STRIPE_LEN;
      int sk = secOff   + s * XXH_SECRET_CONSUME_RATE;
      long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
      long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
      long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
      long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
      long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
      long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
      long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
      long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    if (len > XXH_STRIPE_LEN) {
      int b  = off + len - XXH_STRIPE_LEN;
      int sk = secOff + (int) LAST_ACC_OFFSET;
      long d0 = getLE64VH(in, b),      k0 = d0 ^ getLE64VH(secret, sk);
      long d1 = getLE64VH(in, b + 8),  k1 = d1 ^ getLE64VH(secret, sk + 8);
      long d2 = getLE64VH(in, b + 16), k2 = d2 ^ getLE64VH(secret, sk + 16);
      long d3 = getLE64VH(in, b + 24), k3 = d3 ^ getLE64VH(secret, sk + 24);
      long d4 = getLE64VH(in, b + 32), k4 = d4 ^ getLE64VH(secret, sk + 32);
      long d5 = getLE64VH(in, b + 40), k5 = d5 ^ getLE64VH(secret, sk + 40);
      long d6 = getLE64VH(in, b + 48), k6 = d6 ^ getLE64VH(secret, sk + 48);
      long d7 = getLE64VH(in, b + 56), k7 = d7 ^ getLE64VH(secret, sk + 56);
      a1 += d0;  a0 += (k0 & 0xFFFFFFFFL) * (k0 >>> 32);
      a0 += d1;  a1 += (k1 & 0xFFFFFFFFL) * (k1 >>> 32);
      a3 += d2;  a2 += (k2 & 0xFFFFFFFFL) * (k2 >>> 32);
      a2 += d3;  a3 += (k3 & 0xFFFFFFFFL) * (k3 >>> 32);
      a5 += d4;  a4 += (k4 & 0xFFFFFFFFL) * (k4 >>> 32);
      a4 += d5;  a5 += (k5 & 0xFFFFFFFFL) * (k5 >>> 32);
      a7 += d6;  a6 += (k6 & 0xFFFFFFFFL) * (k6 >>> 32);
      a6 += d7;  a7 += (k7 & 0xFFFFFFFFL) * (k7 >>> 32);
    }

    int ms1 = secOff + XXH_SECRET_MERGEACCS_START;
    long low = XXH3_avalanche(len * XXH_PRIME64_1
      + XXH3_mul128_fold64(a0 ^ getLE64VH(secret, ms1),      a1 ^ getLE64VH(secret, ms1 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64VH(secret, ms1 + 16), a3 ^ getLE64VH(secret, ms1 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64VH(secret, ms1 + 32), a5 ^ getLE64VH(secret, ms1 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64VH(secret, ms1 + 48), a7 ^ getLE64VH(secret, ms1 + 56)));
    int ms2 = secOff + XXH3_SECRET_DEFAULT_SIZE - XXH_STRIPE_LEN - XXH_SECRET_MERGEACCS_START;
    long high = XXH3_avalanche(~(len * XXH_PRIME64_2)
      + XXH3_mul128_fold64(a0 ^ getLE64VH(secret, ms2),      a1 ^ getLE64VH(secret, ms2 + 8))
      + XXH3_mul128_fold64(a2 ^ getLE64VH(secret, ms2 + 16), a3 ^ getLE64VH(secret, ms2 + 24))
      + XXH3_mul128_fold64(a4 ^ getLE64VH(secret, ms2 + 32), a5 ^ getLE64VH(secret, ms2 + 40))
      + XXH3_mul128_fold64(a6 ^ getLE64VH(secret, ms2 + 48), a7 ^ getLE64VH(secret, ms2 + 56)));
    return toByteArray128(high, low);
  }

  // ==========================================================================
  // Custom secret generation. Mirrors XXH3_initCustomSecret_scalar
  // (xxhash.h ~line 6090).
  // ==========================================================================
  private static byte[] XXH3_initCustomSecret(long seed) {
    byte[] cs = new byte[XXH3_SECRET_DEFAULT_SIZE];
    int nbRounds = XXH3_SECRET_DEFAULT_SIZE / 16;
    for (int i = 0; i < nbRounds; i++) {
      long s0 = getLE64(XXH3_kSecret, SEC_BASE + 16L * i)     + seed;
      long s1 = getLE64(XXH3_kSecret, SEC_BASE + 16L * i + 8) - seed;
      putLE64(cs, SEC_BASE + 16L * i,     s0);
      putLE64(cs, SEC_BASE + 16L * i + 8, s1);
    }
    return cs;
  }

  // ==========================================================================
  // Internal dispatchers. Pick the right size bucket and delegate. The
  // long-input branch derives a custom secret from seed when seed != 0.
  // ==========================================================================

  private static long XXH3_64bits_internal(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    if (len <= 16)               return XXH3_len_0to16_64b(in, off, len, secret, secOff, seed);
    if (len <= 128)              return XXH3_len_17to128_64b(in, off, len, secret, secOff, seed);
    if (len <= XXH3_MIDSIZE_MAX) return XXH3_len_129to240_64b(in, off, len, secret, secOff, seed);
    byte[] secretUse = (seed == 0) ? secret : XXH3_initCustomSecret(seed);
    long   secOffUse = (seed == 0) ? secOff : SEC_BASE;
    return XXH3_hashLong_64b(in, off, len, secretUse, secOffUse);
  }

  private static byte[] XXH3_128bits_internal(
      Object in, long off, int len, byte[] secret, long secOff, long seed) {
    if (len <= 16)               return XXH3_len_0to16_128b(in, off, len, secret, secOff, seed);
    if (len <= 128)              return XXH3_len_17to128_128b(in, off, len, secret, secOff, seed);
    if (len <= XXH3_MIDSIZE_MAX) return XXH3_len_129to240_128b(in, off, len, secret, secOff, seed);
    byte[] secretUse = (seed == 0) ? secret : XXH3_initCustomSecret(seed);
    long   secOffUse = (seed == 0) ? secOff : SEC_BASE;
    return XXH3_hashLong_128b(in, off, len, secretUse, secOffUse);
  }

  // ==========================================================================
  // Output encoding
  // ==========================================================================

  /** Encodes 128-bit result as byte[16] in big-endian canonical form: [high64 || low64]. */
  private static byte[] toByteArray128(long high64, long low64) {
    byte[] result = new byte[16];
    putLE64VH(result, 0, Long.reverseBytes(high64));
    putLE64VH(result, 8, Long.reverseBytes(low64));
    return result;
  }

  /**
   * XOR-folds a 16-byte big-endian XXH3 128-bit hash to a single long (high64 XOR low64).
   * Used by the XxHash128 codegen path to chain multi-input seeds without re-decoding
   * the byte[] through ByteBuffer.
   */
  public static long fold128(byte[] hash128) {
    long high = Long.reverseBytes(getLE64VH(hash128, 0));
    long low  = Long.reverseBytes(getLE64VH(hash128, 8));
    return high ^ low;
  }

  // ==========================================================================
  // Public 64-bit API - no-seed variants
  // ==========================================================================

  /** Hash a 32-bit integer using seed 0. */
  public static long hashInt64(int input) { return hashInt64(input, 0L); }

  /** Hash a 64-bit long using seed 0. */
  public static long hashLong64(long input) { return hashLong64(input, 0L); }

  /** Hash a raw byte region from Spark unsafe memory using seed 0. */
  public static long hashUnsafeBytes64(Object base, long offset, int len) {
    return XXH3_64bits_internal(base, offset, len, XXH3_kSecret, SEC_BASE, 0L);
  }

  /** Word-aligned variant of {@link #hashUnsafeBytes64} (len must be a multiple of 8). */
  public static long hashUnsafeWords64(Object base, long offset, int len) {
    if (len % 8 != 0) throw new IllegalArgumentException("len must be a multiple of 8: " + len);
    return hashUnsafeBytes64(base, offset, len);
  }

  /** Hash a UTF-8 string using seed 0. */
  public static long hashUTF8String64(UTF8String str) {
    return hashUnsafeBytes64(str.getBaseObject(), str.getBaseOffset(), str.numBytes());
  }

  // ==========================================================================
  // Public 64-bit API - seeded variants (used by the codegen hash path).
  // ==========================================================================

  /** Hash a 32-bit integer with an explicit seed. */
  public static long hashInt64(int input, long seed) {
    seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
    long i = input & 0xFFFFFFFFL;
    long input64 = i | (i << 32);
    long bitflip = BITFLIP_64_4_8 - seed;
    return XXH3_rrmxmx(input64 ^ bitflip, 4);
  }

  /** Hash a 64-bit long with an explicit seed. */
  public static long hashLong64(long input, long seed) {
    seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
    long bitflip = BITFLIP_64_4_8 - seed;
    return XXH3_rrmxmx(Long.rotateLeft(input, 32) ^ bitflip, 8);
  }

  /** Hash a raw byte region from Spark unsafe memory with an explicit seed. */
  public static long hashUnsafeBytes64(Object base, long offset, int len, long seed) {
    return XXH3_64bits_internal(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
  }

  // ==========================================================================
  // Public 128-bit API - no-seed variants
  // ==========================================================================

  /** 128-bit hash of a raw byte region from Spark unsafe memory using seed 0. */
  public static byte[] hashUnsafeBytes128(Object base, long offset, int len) {
    return XXH3_128bits_internal(base, offset, len, XXH3_kSecret, SEC_BASE, 0L);
  }

  /** 128-bit hash of a UTF-8 string using seed 0. */
  public static byte[] hashUTF8String128(UTF8String str) {
    return hashUnsafeBytes128(str.getBaseObject(), str.getBaseOffset(), str.numBytes());
  }

  /** 128-bit hash of a 32-bit integer using seed 0. */
  public static byte[] hashInt128(int input) {
    long i = input & 0xFFFFFFFFL;
    long in64 = i | (i << 32);
    long keyed = in64 ^ BITFLIP_128_4_8;
    long multiplier = XXH_PRIME64_1 + 16L;
    long mLow  = XXH_mult128_low64(keyed, multiplier);
    long mHigh = XXH_mult128_high64(keyed, multiplier);
    mHigh += mLow << 1;
    mLow  ^= mHigh >>> 3;
    long low  = mLow ^ (mLow >>> 35);
    low  *= PRIME_MX2;
    low  ^= low >>> 28;
    return toByteArray128(XXH3_avalanche(mHigh), low);
  }

  /** 128-bit hash of a 64-bit long using seed 0. */
  public static byte[] hashLong128(long input) {
    long keyed = input ^ BITFLIP_128_4_8;
    long multiplier = XXH_PRIME64_1 + 32L;
    long mLow  = XXH_mult128_low64(keyed, multiplier);
    long mHigh = XXH_mult128_high64(keyed, multiplier);
    mHigh += mLow << 1;
    mLow  ^= mHigh >>> 3;
    long low  = mLow ^ (mLow >>> 35);
    low  *= PRIME_MX2;
    low  ^= low >>> 28;
    return toByteArray128(XXH3_avalanche(mHigh), low);
  }

  // ==========================================================================
  // Public 128-bit API - seeded variants
  // ==========================================================================

  /** 128-bit hash of a raw byte region from Spark unsafe memory with an explicit seed. */
  public static byte[] hashUnsafeBytes128(Object base, long offset, int len, long seed) {
    return XXH3_128bits_internal(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
  }

  /** 128-bit hash of a UTF-8 string with an explicit seed. */
  public static byte[] hashUTF8String128(UTF8String str, long seed) {
    return hashUnsafeBytes128(str.getBaseObject(), str.getBaseOffset(), str.numBytes(), seed);
  }

  /** 128-bit hash of a 32-bit integer with an explicit seed. */
  public static byte[] hashInt128(int input, long seed) {
    byte[] buf = new byte[16];
    hashInt128Into(input, seed, buf);
    return buf;
  }

  /**
   * 128-bit hash of a 32-bit integer into a caller-supplied 16-byte buffer (no allocation).
   * Suitable for codegen hot paths where the buffer is pre-allocated per generated class.
   */
  public static void hashInt128Into(int input, long seed, byte[] buf) {
    seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
    long i = input & 0xFFFFFFFFL;
    long in64 = i | (i << 32);
    long keyed = in64 ^ (BITFLIP_128_4_8 + seed);
    long multiplier = XXH_PRIME64_1 + 16L;
    long mLow  = XXH_mult128_low64(keyed, multiplier);
    long mHigh = XXH_mult128_high64(keyed, multiplier);
    mHigh += mLow << 1;
    mLow  ^= mHigh >>> 3;
    long low  = mLow ^ (mLow >>> 35);
    low  *= PRIME_MX2;
    low  ^= low >>> 28;
    putLE64VH(buf, 0, Long.reverseBytes(XXH3_avalanche(mHigh)));
    putLE64VH(buf, 8, Long.reverseBytes(low));
  }

  /** 128-bit hash of a 64-bit long with an explicit seed. */
  public static byte[] hashLong128(long input, long seed) {
    byte[] buf = new byte[16];
    hashLong128Into(input, seed, buf);
    return buf;
  }

  /**
   * 128-bit hash of a 64-bit long into a caller-supplied 16-byte buffer (no allocation).
   * Suitable for codegen hot paths where the buffer is pre-allocated per generated class.
   */
  public static void hashLong128Into(long input, long seed, byte[] buf) {
    seed ^= ((long) Integer.reverseBytes((int) seed)) << 32;
    long keyed = input ^ (BITFLIP_128_4_8 + seed);
    long multiplier = XXH_PRIME64_1 + 32L;
    long mLow  = XXH_mult128_low64(keyed, multiplier);
    long mHigh = XXH_mult128_high64(keyed, multiplier);
    mHigh += mLow << 1;
    mLow  ^= mHigh >>> 3;
    long low  = mLow ^ (mLow >>> 35);
    low  *= PRIME_MX2;
    low  ^= low >>> 28;
    putLE64VH(buf, 0, Long.reverseBytes(XXH3_avalanche(mHigh)));
    putLE64VH(buf, 8, Long.reverseBytes(low));
  }

  // ==========================================================================
  // byte[]-typed overloads. Same algorithm and output as the Object-typed
  // entries above, but the input is typed byte[] all the way through so the
  // JIT can apply array-specific alias analysis. Java overload resolution
  // picks these at compile time when the caller has a static byte[] type
  // (e.g. codegen, benchmarks calling on raw arrays).
  // ==========================================================================

  /** 64-bit XXH3 hash of a byte[] region using seed 0. */
  public static long hashUnsafeBytes64(byte[] base, long offset, int len) {
    return hashUnsafeBytes64(base, offset, len, 0L);
  }

  /** 64-bit XXH3 hash of a byte[] region with an explicit seed. */
  public static long hashUnsafeBytes64(byte[] base, long offset, int len, long seed) {
    if (len <= 16)
      return XXH3_len_0to16_64b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= 128)
      return XXH3_len_17to128_64b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= XXH3_MIDSIZE_MAX)
      return XXH3_len_129to240_64b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    byte[] secret = (seed == 0) ? XXH3_kSecret : XXH3_initCustomSecret(seed);
    return XXH3_hashLong_64b_ba(base, offset, len, secret, SEC_BASE);
  }

  /** 128-bit XXH3 hash of a byte[] region using seed 0. */
  public static byte[] hashUnsafeBytes128(byte[] base, long offset, int len) {
    return hashUnsafeBytes128(base, offset, len, 0L);
  }

  /** 128-bit XXH3 hash of a byte[] region with an explicit seed. */
  public static byte[] hashUnsafeBytes128(byte[] base, long offset, int len, long seed) {
    if (len <= 16)
      return XXH3_len_0to16_128b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= 128)
      return XXH3_len_17to128_128b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= XXH3_MIDSIZE_MAX)
      return XXH3_len_129to240_128b(base, offset, len, XXH3_kSecret, SEC_BASE, seed);
    byte[] secret = (seed == 0) ? XXH3_kSecret : XXH3_initCustomSecret(seed);
    return XXH3_hashLong_128b_ba(base, offset, len, secret, SEC_BASE);
  }

  // ==========================================================================
  // VarHandle public API for byte[] input with zero-based int index.
  //
  // Short inputs (<=240 bytes) use the same small-input specializations as the
  // Object/Unsafe path. Long inputs use the VarHandle hot loops above.
  //
  // Callers that hold a byte[] at a known zero-based index should prefer these
  // over hashUnsafeBytes64/128(byte[], long, ...) since int offsets avoid the
  // Platform.BYTE_ARRAY_OFFSET addition. Off-heap callers must use the
  // Object-typed hashUnsafeBytes64/128(Object, long, ...) overloads.
  // ==========================================================================

  /** 64-bit XXH3 of a byte[] region [idx, idx+len) using VarHandle reads for long inputs. */
  public static long hashBytes64(byte[] base, int idx, int len) {
    return hashBytes64(base, idx, len, 0L);
  }

  /** 64-bit XXH3 of a byte[] region [idx, idx+len) with an explicit seed. */
  public static long hashBytes64(byte[] base, int idx, int len, long seed) {
    long off = Platform.BYTE_ARRAY_OFFSET + idx;
    if (len <= 16)
      return XXH3_len_0to16_64b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= 128)
      return XXH3_len_17to128_64b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= XXH3_MIDSIZE_MAX)
      return XXH3_len_129to240_64b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    byte[] secret = (seed == 0) ? XXH3_kSecret : XXH3_initCustomSecret(seed);
    return XXH3_hashLong_64b_vh(base, idx, len, secret, 0);
  }

  /** 128-bit XXH3 of a byte[] region [idx, idx+len) using VarHandle reads for long inputs. */
  public static byte[] hashBytes128(byte[] base, int idx, int len) {
    return hashBytes128(base, idx, len, 0L);
  }

  /** 128-bit XXH3 of a byte[] region [idx, idx+len) with an explicit seed. */
  public static byte[] hashBytes128(byte[] base, int idx, int len, long seed) {
    long off = Platform.BYTE_ARRAY_OFFSET + idx;
    if (len <= 16)
      return XXH3_len_0to16_128b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= 128)
      return XXH3_len_17to128_128b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    if (len <= XXH3_MIDSIZE_MAX)
      return XXH3_len_129to240_128b(base, off, len, XXH3_kSecret, SEC_BASE, seed);
    byte[] secret = (seed == 0) ? XXH3_kSecret : XXH3_initCustomSecret(seed);
    return XXH3_hashLong_128b_vh(base, idx, len, secret, 0);
  }

  private XXH3() {}
}
