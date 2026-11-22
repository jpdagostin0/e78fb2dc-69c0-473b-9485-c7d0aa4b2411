pub mod brain;
pub mod nav;
pub mod physics;
pub mod proto;
pub mod runtime;
pub mod world;

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Size multipliers (from cubes) that get their own nav graph besides the base one.
pub const NAV_SCALES: [f64; 3] = [1.2, 1.4, 1.6];

pub struct Shared {
    pub world: world::World,
    pub nav: nav::Nav,
    /// (scale, nav built for a mech of that size), ascending by scale.
    pub big: Vec<(f64, nav::Nav)>,
}

impl Shared {
    /// Never fails: falls back to World::empty().
    pub fn load(map_dir: &std::path::Path, cfg: &proto::Config) -> Shared {
        let world = match catch_unwind(AssertUnwindSafe(|| world::World::load(map_dir))) {
            Ok(Ok(w)) => w,
            Ok(Err(e)) => {
                runtime::log(&format!("world load failed: {e}"));
                world::World::empty()
            }
            Err(_) => {
                runtime::log("world load panicked");
                world::World::empty()
            }
        };
        let nav = match catch_unwind(AssertUnwindSafe(|| nav::Nav::build(&world, &cfg.mech))) {
            Ok(n) => n,
            Err(_) => {
                runtime::log("nav build panicked, using empty world");
                nav::Nav::build(&world::World::empty(), &cfg.mech)
            }
        };
        let mut big = Vec::new();
        for s in NAV_SCALES {
            let mut m = cfg.mech.clone();
            m.radius *= s;
            m.height *= s;
            match catch_unwind(AssertUnwindSafe(|| nav::Nav::build(&world, &m))) {
                Ok(n) => big.push((s, n)),
                Err(_) => runtime::log(&format!("nav build panicked at scale {s}")),
            }
        }
        Shared { world, nav, big }
    }

    /// The nav graph built for a mech `scale` times the base size; the base graph for any other value.
    pub fn nav_at(&self, scale: f64) -> &nav::Nav {
        self.big.iter().find(|(s, _)| *s == scale).map_or(&self.nav, |(_, n)| n)
    }
}

pub trait Pilot: Send {
    fn act(&mut self, obs: &proto::Obs) -> proto::Action;
}
