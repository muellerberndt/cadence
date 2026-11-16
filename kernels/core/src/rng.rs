//! NumPy's `Generator` on `PCG64`, draw for draw: the 128-bit linear congruential step with
//! the XSL-RR output, the 32-bit draws buffered as NumPy buffers them, `random` from 53 bits,
//! `integers` by Lemire's method (NumPy's default, not the masked one) and `choice` with or
//! without probabilities. A generator crosses with NumPy's `bit_generator.state`: the state,
//! the increment, and the buffered half of a draw.

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
}
