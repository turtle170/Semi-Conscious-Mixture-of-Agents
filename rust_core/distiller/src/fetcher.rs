#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct ContextFetcher {
    history: Vec<(Vec<u8>, [f32; 2])>,
}

impl ContextFetcher {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    pub fn record(&mut self, sequence: Vec<u8>, physics: [f32; 2]) {
        self.history.push((sequence, physics));
        if self.history.len() > 10000 {
            self.history.remove(0);
        }
    }

    pub fn query_context(&self, current_sequence: &[u8]) -> [f32; 2] {
        if self.history.is_empty() {
            return [9.8, 0.99]; // Default gravity, friction
        }

        let mut best_match = [9.8, 0.99];
        let mut min_distance = usize::MAX;

        for (hist_seq, physics) in &self.history {
            let distance = self.hamming_distance(hist_seq, current_sequence);
            if distance < min_distance {
                min_distance = distance;
                best_match = *physics;
            }
        }

        best_match
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "avx10.1-256")]
            {
                return unsafe { self.hamming_distance_avx10(a, b) };
            }

            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                return unsafe { self.hamming_distance_avx512(a, b) };
            } else if std::is_x86_feature_detected!("avx2") {
                return unsafe { self.hamming_distance_avx2(a, b) };
            } else if std::is_x86_feature_detected!("avx") {
                return unsafe { self.hamming_distance_avx(a, b) };
            }
        }
        self.hamming_distance_scalar(a, b)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx10.1-256"))]
    #[target_feature(enable = "avx10.1-256")]
    unsafe fn hamming_distance_avx10(&self, a: &[u8], b: &[u8]) -> usize {
        self.hamming_distance_avx2(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn hamming_distance_avx512(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;

        while i + 64 <= len {
            let chunk_a = _mm512_loadu_si512(a[i..].as_ptr() as *const _);
            let chunk_b = _mm512_loadu_si512(b[i..].as_ptr() as *const _);
            let mask = _mm512_cmpneq_epi8_mask(chunk_a, chunk_b);
            dist += mask.count_ones() as usize;
            i += 64;
        }

        while i < len {
            if a[i] != b[i] {
                dist += 1;
            }
            i += 1;
        }

        dist + (a.len().abs_diff(b.len()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hamming_distance_avx2(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;

        while i + 32 <= len {
            let chunk_a = _mm256_loadu_si256(a[i..].as_ptr() as *const _);
            let chunk_b = _mm256_loadu_si256(b[i..].as_ptr() as *const _);
            let eq = _mm256_cmpeq_epi8(chunk_a, chunk_b);
            let mask = _mm256_movemask_epi8(eq);
            dist += 32 - mask.count_ones() as usize;
            i += 32;
        }

        while i < len {
            if a[i] != b[i] {
                dist += 1;
            }
            i += 1;
        }

        dist + (a.len().abs_diff(b.len()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn hamming_distance_avx(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;

        while i + 16 <= len {
            let chunk_a = _mm_loadu_si128(a[i..].as_ptr() as *const _);
            let chunk_b = _mm_loadu_si128(b[i..].as_ptr() as *const _);
            let eq = _mm_cmpeq_epi8(chunk_a, chunk_b);
            let mask = _mm_movemask_epi8(eq);
            dist += 16 - mask.count_ones() as usize;
            i += 16;
        }

        while i < len {
            if a[i] != b[i] {
                dist += 1;
            }
            i += 1;
        }

        dist + (a.len().abs_diff(b.len()))
    }

    fn hamming_distance_scalar(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        for i in 0..len {
            if a[i] != b[i] {
                dist += 1;
            }
        }
        dist + (a.len().abs_diff(b.len()))
    }
}
