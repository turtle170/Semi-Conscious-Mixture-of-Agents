use env::PhysicsParams;

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
            let distance = self.hamming_distance(hist_seq, current_sequence);
            if distance < min_distance {
                min_distance = distance;
                best_match = *params;
            }
        }

        best_match
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
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
