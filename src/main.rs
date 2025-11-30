#![feature(portable_simd)]
#![feature(cold_path)]
#![feature(slice_split_once)]
#![feature(hasher_prefixfree_extras)]
#![feature(ptr_cast_array)]

use std::io::Write;
use std::{
    borrow::Borrow,
    collections::{BTreeMap, btree_map::Entry},
    ffi::{c_int, c_void},
    fs::File,
    hash::{Hash, Hasher},
    os::fd::AsRawFd,
    simd::{cmp::SimdPartialEq, u8x64},
};

type HashMap<K, V> = std::collections::HashMap<K, V, hasher::FastHasherBuilder>;

const SEMI: u8x64 = u8x64::splat(b';');
const NEWL: u8x64 = u8x64::splat(b'\n');

const INLINE: usize = 16;
const LAST: usize = INLINE - 1;

union StrVec {
    inlined: [u8; INLINE],
    // if length high bit is set, then inlined into pointer then len
    // otherwise, pointer is a pointer to Vec<u8>
    heap: (usize, *mut u8),
}

// SAFETY: effectively just a Vec<str>, which is fine across thread boundaries
unsafe impl Send for StrVec {}

impl StrVec {
    pub fn new(s: &[u8]) -> Self {
        if s.len() < INLINE {
            let mut combined = [0u8; INLINE];
            combined[..s.len()].copy_from_slice(s);
            combined[LAST] = s.len() as u8 + 1;
            Self { inlined: combined }
        } else {
            let ptr = Box::into_raw(s.to_vec().into_boxed_slice());
            Self {
                heap: (ptr.len().to_be(), ptr as *mut u8),
            }
        }
    }
}

impl Drop for StrVec {
    fn drop(&mut self) {
        if unsafe { self.inlined[LAST] } == 0x00 {
            unsafe {
                let len = usize::from_be(self.heap.0);
                let ptr = self.heap.1;
                let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, len);
                let _ = Box::from_raw(slice_ptr);
            }
        }
    }
}

impl AsRef<[u8]> for StrVec {
    fn as_ref(&self) -> &[u8] {
        unsafe {
            if self.inlined[LAST] != 0x00 {
                let len = self.inlined[LAST] as usize - 1;
                std::slice::from_raw_parts(self.inlined.as_ptr(), len)
            } else {
                std::hint::cold_path();
                let len = usize::from_be(self.heap.0);
                let ptr = self.heap.1;
                std::slice::from_raw_parts(ptr, len)
            }
        }
    }
}

impl PartialEq for StrVec {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            self.inlined[LAST] == other.inlined[LAST] && {
                std::hint::cold_path();
                self.as_ref() == other.as_ref()
            }
        }
    }
}

impl Eq for StrVec {}

impl Hash for StrVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl Borrow<[u8]> for StrVec {
    fn borrow(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Debug, Clone, Copy)]
struct Stat {
    min: i16,
    max: i16,
    sum: i64,
    count: u32,
}

impl Default for Stat {
    fn default() -> Self {
        Self {
            min: i16::MAX,
            sum: 0,
            count: 0,
            max: i16::MIN,
        }
    }
}

fn main() {
    let start = std::time::Instant::now();
    let f = File::open("measurements.txt").unwrap();
    let mut stats = BTreeMap::new();
    std::thread::scope(|scope| {
        let map = mmap(&f);
        let nthreads = std::thread::available_parallelism().unwrap();
        let mut at = 0;
        let (tx, rx) = std::sync::mpsc::sync_channel(nthreads.get());
        let chunk_size = map.len() / nthreads;
        for _ in 0..nthreads.get() {
            let start = at;
            let end = (at + chunk_size).min(map.len());
            let end = if end == map.len() {
                map.len()
            } else {
                let newline_at = next_newline(&map[end..], 0);
                end + newline_at + 1
            };
            let map = &map[start..end];
            at = end;
            let tx = tx.clone();
            scope.spawn(move || tx.send(one(map)));
        }

        drop(tx);
        for one_stat in rx {
            for (k, v) in one_stat {
                // SAFETY: the README promised
                match stats.entry(unsafe { String::from_utf8_unchecked(k.as_ref().to_vec()) }) {
                    Entry::Vacant(none) => {
                        none.insert(v);
                    }
                    Entry::Occupied(some) => {
                        let stat = some.into_mut();
                        stat.min = stat.min.min(v.min);
                        stat.sum += v.sum;
                        stat.count += v.count;
                        stat.max = stat.max.max(v.max);
                    }
                }
            }
        }
    });

    print(stats);
    eprintln!("\n{:?}", start.elapsed());
}

#[inline(never)]
fn print(stats: BTreeMap<String, Stat>) {
    let stdout = std::io::stdout();
    let stdout = stdout.lock();
    let mut writer = std::io::BufWriter::new(stdout);
    write!(writer, "{{").unwrap();
    let stats = BTreeMap::from_iter(
        stats
            .iter()
            // SAFETY: the README promised
            .map(|(k, v)| (unsafe { std::str::from_utf8_unchecked(k.as_ref()) }, *v)),
    );
    let mut stats = stats.into_iter().peekable();
    while let Some((station, stat)) = stats.next() {
        write!(
            writer,
            "{station}={:.1}/{:.1}/{:.1}",
            (stat.min as f64) / 10.,
            (stat.sum as f64) / 10. / (stat.count as f64),
            (stat.max as f64) / 10.
        )
        .unwrap();
        if stats.peek().is_some() {
            write!(writer, ", ").unwrap();
        }
    }
    write!(writer, "}}").unwrap();
}

#[inline(never)]
fn one(map: &[u8]) -> HashMap<StrVec, Stat> {
    let mut stats = HashMap::with_capacity_and_hasher(1_024, Default::default());
    let mut at = 0;
    while at < map.len() {
        let newline_at = at + next_newline(map, at);
        let line = unsafe { map.get_unchecked(at..newline_at) };
        at = newline_at + 1;
        let semi = semi_at(line);
        let station = unsafe { line.get_unchecked(..semi) };
        let temperature = unsafe { line.get_unchecked(semi + 1..) };
        let t = parse_temperature(temperature);
        update_stats(&mut stats, station, t);
    }
    stats
}

fn update_stats(stats: &mut HashMap<StrVec, Stat>, station: &[u8], t: i16) {
    let stats = match stats.get_mut(station) {
        Some(stats) => stats,
        None => stats.entry(StrVec::new(station)).or_default(),
    };
    if t < stats.min {
        stats.min = t;
    }
    if t > stats.max {
        stats.max = t;
    }
    stats.sum += i64::from(t);
    stats.count += 1;
}

#[inline]
fn next_newline(map: &[u8], at: usize) -> usize {
    let rest = unsafe { map.get_unchecked(at..) };
    let against = if let Some(restu8x64) = rest.first_chunk::<64>() {
        u8x64::from_array(*restu8x64)
    } else {
        std::hint::cold_path();
        u8x64::load_or_default(rest)
    };
    let newline_eq = NEWL.simd_eq(against);
    if let Some(i) = newline_eq.first_set() {
        i
    } else {
        // we know, line is at most 100+1+5 = 106b,
        // but we can only search 64b, so the search _may_ have to fall back to memchr
        // we know there _must_ be a newline, so rest[64..] must be non-empty
        std::hint::cold_path();
        let restrest = unsafe { rest.get_unchecked(64..) };
        // SAFETY: restrest is valid for at least restrest.len() bytes
        let next_newline = unsafe {
            libc::memchr(
                restrest.as_ptr() as *const c_void,
                b'\n' as c_int,
                restrest.len(),
            )
        };
        assert!(!next_newline.is_null());
        // SAFETY: memchr always returns pointers in restrest, which are valid
        let len = unsafe { (next_newline as *const u8).offset_from(restrest.as_ptr()) } as usize;
        64 + len
    }
}

#[inline]
fn semi_at(line: &[u8]) -> usize {
    // we know, line is at most 100+1+5 = 106b
    if line.len() > 64 {
        std::hint::cold_path();
        line.iter().position(|c| *c == b';').unwrap()
    } else {
        let delim_eq = SEMI.simd_eq(u8x64::load_or_default(line));
        // SAFETY: we're promised there is a ; in every line
        unsafe { delim_eq.first_set().unwrap_unchecked() }
    }
}

#[inline]
fn parse_temperature(t: &[u8]) -> i16 {
    let tlen = t.len();
    unsafe { std::hint::assert_unchecked(tlen >= 3) };
    let is_neg = std::hint::select_unpredictable(t[0] == b'-', true, false);
    let sign = i16::from(!is_neg) * 2 - 1;
    let skip = usize::from(is_neg);
    let has_dd = std::hint::select_unpredictable(tlen - skip == 4, true, false);
    let mul = i16::from(has_dd) * 90 + 10;
    let t1 = mul * i16::from(t[skip] - b'0');
    let t2 = i16::from(has_dd) * 10 * i16::from(t[tlen - 3] - b'0');
    let t3 = i16::from(t[tlen - 1] - b'0');
    sign * (t1 + t2 + t3)
}

#[test]
fn pt() {
    assert_eq!(parse_temperature(b"0.0"), 0);
    assert_eq!(parse_temperature(b"9.2"), 92);
    assert_eq!(parse_temperature(b"-9.2"), -92);
    assert_eq!(parse_temperature(b"98.2"), 982);
    assert_eq!(parse_temperature(b"-98.2"), -982);
}

fn mmap(f: &File) -> &'_ [u8] {
    let len = f.metadata().unwrap().len();
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len as libc::size_t,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            0,
        );

        if ptr == libc::MAP_FAILED {
            panic!("{:?}", std::io::Error::last_os_error());
        } else {
            if libc::madvise(ptr, len as libc::size_t, libc::MADV_SEQUENTIAL) != 0 {
                panic!("{:?}", std::io::Error::last_os_error())
            }
            std::slice::from_raw_parts(ptr as *const u8, len as usize)
        }
    }
}

/// This code is adapted from (and is a strictly less general version of) [foldhash](https://docs.rs/foldhash/).
mod hasher {
    use std::hash::Hasher;

    #[derive(Default)]
    pub struct FastHasher {
        accumulator: u64,
    }

    pub type FastHasherBuilder = std::hash::BuildHasherDefault<FastHasher>;

    #[inline(always)]
    fn folded_multiply(x: u64, y: u64) -> u64 {
        // We compute the full u64 x u64 -> u128 product, this is a single mul
        // instruction on x86-64, one mul plus one mulhi on ARM64.
        let full = (x as u128).wrapping_mul(y as u128);
        let lo = full as u64;
        let hi = (full >> 64) as u64;

        // The middle bits of the full product fluctuate the most with small
        // changes in the input. This is the top bits of lo and the bottom bits
        // of hi. We can thus make the entire output fluctuate with small
        // changes to the input by XOR'ing these two halves.
        lo ^ hi
    }

    impl FastHasher {
        const FIXED_SEED: [u64; 2] = [0xc0ac29b7c97c50dd, 0x3f84d5b5b5470917];
        #[inline(always)]
        fn hash_bytes_short(bytes: &[u8], accumulator: u64) -> u64 {
            let len = bytes.len();
            let mut s0 = accumulator;
            let mut s1 = Self::FIXED_SEED[1];
            // XOR the input into s0, s1, then multiply and fold.
            if len >= 8 {
                s0 ^= u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
                s1 ^= u64::from_ne_bytes(bytes[len - 8..].try_into().unwrap());
            } else if len >= 4 {
                s0 ^= u32::from_ne_bytes(bytes[0..4].try_into().unwrap()) as u64;
                s1 ^= u32::from_ne_bytes(bytes[len - 4..].try_into().unwrap()) as u64;
            } else if len > 0 {
                let lo = bytes[0];
                let mid = bytes[len / 2];
                let hi = bytes[len - 1];
                s0 ^= lo as u64;
                s1 ^= ((hi as u64) << 8) | mid as u64;
            }
            folded_multiply(s0, s1)
        }

        #[cold]
        #[inline(never)]
        // SAFETY: v.len() must be > 16
        unsafe fn hash_bytes_long(v: &[u8], accumulator: u64) -> u64 {
            debug_assert!(v.len() > 16);
            let mut s0 = accumulator;
            let mut s1 = s0.wrapping_add(Self::FIXED_SEED[1]);
            // for the purposes of this challenge, this can't happen
            if v.len() > 128 {
                unreachable!();
            }

            let len = v.len();
            unsafe {
                // SAFETY: our precondition ensures our length is at least 16, and the
                // above loops do not reduce the length under that. This protects our
                // first iteration of this loop, the further iterations are protected
                // directly by the checks on len.
                s0 = folded_multiply(load(v, 0) ^ s0, load(v, len - 16) ^ Self::FIXED_SEED[0]);
                s1 = folded_multiply(load(v, 8) ^ s1, load(v, len - 8) ^ Self::FIXED_SEED[0]);
                if len >= 32 {
                    std::hint::cold_path();
                    s0 = folded_multiply(load(v, 16) ^ s0, load(v, len - 32) ^ Self::FIXED_SEED[0]);
                    s1 = folded_multiply(load(v, 24) ^ s1, load(v, len - 24) ^ Self::FIXED_SEED[0]);
                    if len >= 64 {
                        std::hint::cold_path();
                        s0 = folded_multiply(
                            load(v, 32) ^ s0,
                            load(v, len - 48) ^ Self::FIXED_SEED[0],
                        );
                        s1 = folded_multiply(
                            load(v, 40) ^ s1,
                            load(v, len - 40) ^ Self::FIXED_SEED[0],
                        );
                        if len >= 96 {
                            std::hint::cold_path();
                            s0 = folded_multiply(
                                load(v, 48) ^ s0,
                                load(v, len - 64) ^ Self::FIXED_SEED[0],
                            );
                            s1 = folded_multiply(
                                load(v, 56) ^ s1,
                                load(v, len - 56) ^ Self::FIXED_SEED[0],
                            );
                        }
                    }
                }
            }
            s0 ^ s1
        }
    }

    impl Hasher for FastHasher {
        #[inline(always)]
        fn write(&mut self, bytes: &[u8]) {
            //self.accumulator = self.accumulator.rotate_right(len as u32);
            if bytes.len() <= 16 {
                self.accumulator = Self::hash_bytes_short(bytes, self.accumulator);
            } else {
                // SAFETY: we checked that len > 16
                unsafe { self.accumulator = Self::hash_bytes_long(bytes, self.accumulator) }
            }
        }

        #[inline(always)]
        fn write_length_prefix(&mut self, _len: usize) {}

        #[inline(always)]
        fn finish(&self) -> u64 {
            self.accumulator
        }
    }

    /// Load 8 bytes into a u64 word at the given offset.
    ///
    /// # Safety
    /// You must ensure that offset + 8 <= bytes.len().
    #[inline(always)]
    unsafe fn load(bytes: &[u8], offset: usize) -> u64 {
        // In most (but not all) cases this unsafe code is not necessary to avoid
        // the bounds checks in the below code, but the register allocation became
        // worse if I replaced those calls which could be replaced with safe code.
        unsafe { bytes.as_ptr().add(offset).cast::<u64>().read_unaligned() }
    }
}
