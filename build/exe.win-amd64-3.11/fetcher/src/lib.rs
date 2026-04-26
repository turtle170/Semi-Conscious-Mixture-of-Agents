use env::PhysicsParams;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct ContextFetcher {
    history: Vec<(Vec<u8>, PhysicsParams)>,
}

impl ContextFetcher {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    pub fn record(&mut self, sequence: Vec<u8>, params: PhysicsParams) {
        self.history.push((sequence, params));
        if self.history.len() > 10000 {
            self.history.remove(0);
        }
    }

    pub fn query_context(&self, current_sequence: &[u8]) -> PhysicsParams {
        if self.history.is_empty() {
            return PhysicsParams { gravity: 9.8, friction: 0.99 };
        }

        let mut best_match = self.history[0].1;
        let mut min_distance = usize::MAX;

        for (hist_seq, params) in &self.history {
            let distance = self.optimized_distance(hist_seq, current_sequence);
            if distance < min_distance {
                min_distance = distance;
                best_match = *params;
            }
        }

        best_match
    }

    #[inline(always)]
    fn optimized_distance(&self, a: &[u8], b: &[u8]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            // HYPER-OPTIMIZATION: CPU Feature Branching for Project Aether
            
            // 1. AVX-512 / AVX-10 Path
            // (AVX-10 256/512 is handled via avx512 features in current LLVM)
            if is_x86_feature_detected!("avx512bw") && a.len() >= 64 {
                return unsafe { self.hamming_avx512(a, b) };
            }
            
            // 2. AVX-VNNI / AVX2 Path
            // Note: AVX-VNNI is for dot-products, for Hamming we stick to AVX2 bitwise ops.
            if is_x86_feature_detected!("avx2") && a.len() >= 32 {
                return unsafe { self.hamming_avx2(a, b) };
            }
            
            // 3. Legacy AVX (Standard i5/i7 compatibility)
            if is_x86_feature_detected!("avx") && a.len() >= 16 {
                return unsafe { self.hamming_avx(a, b) };
            }
        }
        
        // 4. Non-AVX Fallback (Legacy i3 / Mobile / Non-x86)
        self.hamming_fallback(a, b)
    }

    fn hamming_fallback(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        for i in 0..len {
            if a[i] != b[i] {
                dist += 1;
            }
        }
        dist + (a.len().abs_diff(b.len()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hamming_avx2(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;
        
        while i + 32 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(va, vb);
            let mask = _mm256_movemask_epi8(cmp);
            // mask has 1 bit set for each matching byte.
            // Hamming distance is count of non-matching.
            dist += 32 - (mask as u32).count_ones() as usize;
            i += 32;
        }
        dist + self.hamming_fallback(&a[i..], &b[i..])
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn hamming_avx(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;
        
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(va, vb);
            let mask = _mm_movemask_epi8(cmp);
            dist += 16 - (mask as u32).count_ones() as usize;
            i += 16;
        }
        dist + self.hamming_fallback(&a[i..], &b[i..])
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw")]
    unsafe fn hamming_avx512(&self, a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut dist = 0;
        let mut i = 0;
        
        while i + 64 <= len {
            let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
            // Compare bytes for equality. Returns a 64-bit mask.
            let mask = _mm512_cmpeq_epu8_mask(va, vb);
            dist += 64 - mask.count_ones() as usize;
            i += 64;
        }
        dist + self.hamming_fallback(&a[i..], &b[i..])
    }
}
