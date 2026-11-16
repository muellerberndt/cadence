//! NumPy's `Generator` on `PCG64`, draw for draw: the 128-bit linear congruential step with
//! the XSL-RR output, the 32-bit draws buffered as NumPy buffers them, `random` from 53 bits,
//! `integers` by Lemire's method (NumPy's default, not the masked one), `choice` with or
//! without probabilities, `permutation` (the shuffle's masked draws), `standard_normal` and
//! `normal` (NumPy's ziggurat, its tables in `ziggurat`), and `uniform`. A generator crosses
//! with NumPy's `bit_generator.state`: the state, the increment, and the buffered half of a draw.

use crate::np;
use crate::ziggurat::{FI, KI, NOR_INV_R, NOR_R, WI};

const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    pub state: u128,
    pub inc: u128,
    pub has_uint32: bool,
    pub uinteger: u32,
}

impl Pcg64 {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let s = self.state;
        let xored = ((s >> 64) as u64) ^ (s as u64);
        xored.rotate_right((s >> 122) as u32)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xffff_ffff) as u32
    }

    /// `random()`: a double in [0, 1) from the top 53 bits of a draw.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `integers(low, high)`, `high` excluded, as NumPy draws one int64 (and each of many).
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        let rng = (high as i128 - 1 - low as i128) as u64;
        low.wrapping_add(self.bounded(rng) as i64)
    }

    /// `random_bounded_uint64` without the mask: a value in [0, rng].
    fn bounded(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            0
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                return self.next_u32() as u64;
            }
            let excl = (rng as u32).wrapping_add(1);
            let mut m = (self.next_u32() as u64) * (excl as u64);
            let mut leftover = (m & 0xFFFF_FFFF) as u32;
            if leftover < excl {
                let threshold = (u32::MAX - rng as u32) % excl;
                while leftover < threshold {
                    m = (self.next_u32() as u64) * (excl as u64);
                    leftover = (m & 0xFFFF_FFFF) as u32;
                }
            }
            m >> 32
        } else if rng == u64::MAX {
            self.next_u64()
        } else {
            let excl = rng + 1;
            let mut m = (self.next_u64() as u128) * (excl as u128);
            let mut leftover = m as u64;
            if leftover < excl {
                let threshold = (u64::MAX - rng) % excl;
                while leftover < threshold {
                    m = (self.next_u64() as u128) * (excl as u128);
                    leftover = m as u64;
                }
            }
            (m >> 64) as u64
        }
    }

    /// `choice(n, p=p)` for one draw: the cumulative sum of `p` scaled by its last entry, one
    /// `random()`, and the first entry past it (`searchsorted(side="right")`).
    pub fn choice_p(&mut self, p: &[f64]) -> usize {
        let mut cdf = Vec::with_capacity(p.len());
        let mut run = 0.0f64;
        for &x in p {
            run += x;
            cdf.push(run);
        }
        let last = *cdf.last().unwrap_or(&1.0);
        for c in cdf.iter_mut() {
            *c /= last;
        }
        let u = self.random();
        cdf.partition_point(|&c| c <= u)
    }

    /// `choice(n)` for one draw: a uniform index, as `integers(0, n)` draws it.
    pub fn choice(&mut self, n: usize) -> usize {
        self.bounded(n as u64 - 1) as usize
    }

    /// `random_interval(max)`: a value in [0, max], drawn under the smallest mask at or above
    /// `max` until it fits (a 32-bit draw, buffered, where `max` fits 32 bits): the draws of
    /// NumPy's `shuffle`.
    pub fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xFFFF_FFFF {
            loop {
                let value = self.next_u32() as u64 & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `permutation(n)`: `arange(n)` shuffled as NumPy's `shuffle` does it, from the last place
    /// down to the second, each swapped with a place drawn by `interval`.
    pub fn permutation(&mut self, n: usize) -> Vec<i64> {
        let mut out: Vec<i64> = (0..n as i64).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u64) as usize;
            out.swap(i, j);
        }
        out
    }

    /// `standard_normal()`: NumPy's ziggurat (`random_standard_normal`): one 64-bit draw gives
    /// the strip (its low 8 bits), the sign and 52 bits; past the strip's bound, the tail (strip
    /// 0, by the C library's `log1p`) or the wedge (by its `exp`) draws again.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -NOR_INV_R * np::log1p(-self.random());
                    let yy = -np::log1p(-self.random());
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 { -(NOR_R + xx) } else { NOR_R + xx };
                    }
                }
            } else if (FI[idx - 1] - FI[idx]) * self.random() + FI[idx] < np::exp(-0.5 * x * x) {
                return x;
            }
        }
    }

    /// `normal(loc, scale)`: `loc + scale * standard_normal()`, as NumPy's C compiles it
    /// (`muladd`).
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        muladd(scale, self.standard_normal(), loc)
    }

    /// `uniform(low, high)`: `low + (high - low) * random()`, as NumPy's C compiles it
    /// (`muladd`).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let range = high - low;
        muladd(range, self.random(), low)
    }
}

/// `c + a * b` as NumPy's own C (`random_normal`, `random_uniform`) is compiled on each machine:
/// fused into one rounding in NumPy's macOS build on ARM (clang contracts it to `fmadd`); two
/// roundings in its Linux builds (ARM and x86 alike) and under Pyodide. Checked against NumPy
/// 2.5.5 on an Apple M-series Mac and on an ARM Linux machine (`check/primitives.py`).
#[inline]
fn muladd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        c + a * b
    }
}

// NumPy's `SeedSequence` (Melissa O'Neill's hash mixing, 32-bit words, a pool of four)
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

impl Pcg64 {
    /// `np.random.default_rng(entropy)` for a sequence of nonnegative integers (a tuple such as
    /// `(seed, draws)`): each integer as its 32-bit words from the lowest (0 as one word), the
    /// words mixed into `SeedSequence`'s pool, four 64-bit words drawn from it, and PCG64 seeded
    /// with the first two as its state and the last two as its stream.
    pub fn from_entropy(entropy: &[u64]) -> Pcg64 {
        let mut words: Vec<u32> = Vec::new();
        for &n in entropy {
            if n == 0 {
                words.push(0);
            }
            let mut n = n;
            while n > 0 {
                words.push(n as u32);
                n >>= 32;
            }
        }
        let mut pool = [0u32; POOL];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for src in 0..POOL {
            for dst in 0..POOL {
                if src != dst {
                    pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hash_const));
                }
            }
        }
        for &word in words.iter().skip(POOL) {
            for slot in pool.iter_mut() {
                *slot = mix(*slot, hashmix(word, &mut hash_const));
            }
        }
        // `generate_state(4, np.uint64)`: eight 32-bit words, read two at a time, low first
        let mut hash_const = INIT_B;
        let mut state = [0u64; 4];
        for i in 0..8 {
            let mut value = pool[i % POOL] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^= value >> XSHIFT;
            state[i / 2] |= (value as u64) << (32 * (i % 2));
        }
        let initstate = ((state[0] as u128) << 64) | state[1] as u128;
        let initseq = ((state[2] as u128) << 64) | state[3] as u128;
        // `pcg_setseq_128_srandom_r`
        let mut g = Pcg64 { state: 0, inc: (initseq << 1) | 1, has_uint32: false, uinteger: 0 };
        g.next_u64();
        g.state = g.state.wrapping_add(initstate);
        g.next_u64();
        g
    }
}
