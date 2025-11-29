#![feature(portable_simd)]
#![feature(slice_split_once)]

use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    ffi::{c_int, c_void},
    fs::File,
    hash::{BuildHasher, Hash, Hasher},
    os::fd::AsRawFd,
    simd::{cmp::SimdPartialEq, u8x64},
};

const SEMI: u8x64 = u8x64::splat(b';');

struct FastHasherBuilder;
struct FastHasher(u64);

impl BuildHasher for FastHasherBuilder {
    type Hasher = FastHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FastHasher(0xcbf29ce484222325)
    }
}

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        let mut last = [1u8; 8];
        (last[..remainder.len()]).copy_from_slice(remainder);
        for &chunk in chunks.iter().chain(std::iter::once(&last)) {
            let mixed = self.0 as u128 * (u64::from_ne_bytes(chunk) as u128);
            self.0 = (mixed >> 64) as u64 ^ mixed as u64;
        }
    }
}

const INLINE: usize = 16;
const LAST: usize = INLINE - 1;

union StrVec {
    inlined: [u8; INLINE],
    // if length high bit is set, then inlined into pointer then len
    // otherwise, pointer is a pointer to Vec<u8>
    heap: (usize, *mut u8),
}

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
                // &self.inlined[..len]
                std::slice::from_raw_parts(self.inlined.as_ptr(), len)
            } else {
                let len = usize::from_be(self.heap.0);
                let ptr = self.heap.1;
                std::slice::from_raw_parts(ptr, len)
            }
        }
    }
}

impl PartialEq for StrVec {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
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

fn main() {
    let f = File::open("measurements.txt").unwrap();
    let map = mmap(&f);
    // TODO: maybe make the key &[u8], but measure since we're breaking MADV_SEQUENTIAL
    let mut stats = HashMap::<StrVec, (i16, i64, usize, i16), _>::with_capacity_and_hasher(
        100_000,
        FastHasherBuilder,
    );
    let mut at = 0;
    loop {
        let line = next_line(map, &mut at);
        if line.is_empty() {
            break;
        }
        let (station, temperature) = split_semi(line);
        let t = parse_temperature(temperature);
        let stats = match stats.get_mut(station) {
            Some(stats) => stats,
            None => stats
                .entry(StrVec::new(station))
                .or_insert((i16::MAX, 0, 0, i16::MIN)),
        };
        stats.0 = stats.0.min(t);
        stats.1 += i64::from(t);
        stats.2 += 1;
        stats.3 = stats.3.max(t);
    }
    print!("{{");
    let stats = BTreeMap::from_iter(
        stats
            .iter()
            // SAFETY: the README promised
            .map(|(k, v)| (unsafe { std::str::from_utf8_unchecked(k.as_ref()) }, *v)),
    );
    let mut stats = stats.into_iter().peekable();
    while let Some((station, (min, sum, count, max))) = stats.next() {
        print!(
            "{station}={:.1}/{:.1}/{:.1}",
            (min as f64) / 10.,
            (sum as f64) / 10. / (count as f64),
            (max as f64) / 10.
        );
        if stats.peek().is_some() {
            print!(", ");
        }
    }
    print!("}}");
}

#[inline]
fn next_line<'a>(map: &'a [u8], at: &mut usize) -> &'a [u8] {
    let rest = &map[*at..];
    // SAFETY: rest is valid for at least rest.len() bytes
    let next_newline =
        unsafe { libc::memchr(rest.as_ptr() as *const c_void, b'\n' as c_int, rest.len()) };
    if next_newline.is_null() {
        // don't need to remember to break, since next iteration will find empty line
        *at += rest.len();
        rest
    } else {
        // SAFETY: memchr always returns pointers in rest, which are valid
        let len = unsafe { (next_newline as *const u8).offset_from(rest.as_ptr()) } as usize;
        *at += len + 1;
        &rest[..len]
    }
}

fn split_semi(line: &[u8]) -> (&[u8], &[u8]) {
    // we know, line is at most 100+1+5 = 106b
    if line.len() > 64 {
        line.rsplit_once(|c| *c == b';').unwrap()
    } else {
        let delim_eq = SEMI.simd_eq(u8x64::load_or_default(line));
        // SAFETY: we're promised there is a ; in every line
        let index_of_delim = unsafe { delim_eq.first_set().unwrap_unchecked() };
        (&line[..index_of_delim], &line[index_of_delim + 1..])
    }
}

fn parse_temperature(temperature: &[u8]) -> i16 {
    let mut t: i16 = 0;
    let mut mul = 1;
    for &d in temperature.iter().rev() {
        match d {
            b'.' => {
                continue;
            }
            b'-' => {
                t = -t;
                break;
            }
            _ => {
                t += i16::from(d - b'0') * mul;
                mul *= 10;
            }
        }
    }
    t
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
