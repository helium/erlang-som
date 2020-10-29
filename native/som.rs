use ndarray::{Array1, Array2};
use rusticsom::{
    default_decay_fn, exponential_decay_fn, gaussian, mh_neighborhood, DecayFn, NeighbourhoodFn,
    SOM,
};
use rustler::resource::ResourceArc;
use rustler::types::atom::ok;
use rustler::{Decoder, Encoder, NifResult, Term};
use std::collections::HashMap;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(PartialEq, Clone, Debug)]
pub struct SOMOptions {
    pub learning_rate: Option<f32>,
    pub sigma: Option<f32>,
    pub decay_fn: Option<DecayFn>,
    pub neighbourhood_fn: Option<NeighbourhoodFn>,
    pub classes: Option<HashMap<String, f64>>,
    pub custom_weighting: bool,
}

impl Default for SOMOptions {
    fn default() -> SOMOptions {
        SOMOptions {
            learning_rate: None,
            sigma: None,
            decay_fn: None,
            neighbourhood_fn: None,
            classes: None,
            custom_weighting: false,
        }
    }
}

impl<'a> Decoder<'a> for SOMOptions {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let mut opts = Self::default();
        use rustler::{Error, MapIterator};
        for (key, value) in MapIterator::new(term).ok_or(Error::BadArg)? {
            if atom::learning_rate() == key {
                opts.learning_rate = Some(value.decode()?)
            } else if atom::sigma() == key {
                opts.sigma = Some(value.decode()?)
            } else if atom::decay_fn() == key {
                opts.decay_fn = if atom::default() == value {
                    Some(default_decay_fn)
                } else if atom::exponential() == value {
                    Some(exponential_decay_fn)
                } else {
                    return Err(Error::BadArg);
                }
            } else if atom::neighbourhood_fn() == key {
                opts.neighbourhood_fn = if atom::gaussian() == value {
                    Some(gaussian)
                } else if atom::mexican_hat() == value {
                    Some(mh_neighborhood)
                } else {
                    return Err(Error::BadArg);
                }
            } else if atom::classes() == key {
                let mut classes: HashMap<String, f64> = HashMap::new();
                for (class, weight) in MapIterator::new(value).ok_or(Error::BadArg)? {
                    classes.insert(class.decode()?, weight.decode()?);
                }
                opts.classes = Some(classes);
            } else {
                return Err(Error::BadArg);
            }
        }
        Ok(opts)
    }
}

////////////////////////////////////////////////////////////////////////////
// Atoms                                                                  //
////////////////////////////////////////////////////////////////////////////

mod atom {
    rustler::atoms! {
        classes,
        decay_fn,
        default,
        exponential,
        gaussian,
        learning_rate,
        mexican_hat,
        neighbourhood_fn,
        sigma,
    }
}

////////////////////////////////////////////////////////////////////////////
// Resource                                                               //
////////////////////////////////////////////////////////////////////////////

#[repr(transparent)]
struct SomResource(RwLock<SOM>);

impl SomResource {
    fn read(&self) -> RwLockReadGuard<'_, SOM> {
        self.0.read().unwrap()
    }

    fn write(&self) -> RwLockWriteGuard<'_, SOM> {
        self.0.write().unwrap()
    }
}

impl From<SOM> for SomResource {
    fn from(other: SOM) -> Self {
        SomResource(RwLock::new(other))
    }
}

type Rsc = rustler::ResourceArc<SomResource>;

////////////////////////////////////////////////////////////////////////////
// NIFs                                                                   //
////////////////////////////////////////////////////////////////////////////

#[rustler::nif]
fn new<'a>(
    env: rustler::Env<'a>,
    length: usize,
    breadth: usize,
    inputs: usize,
    randomize: bool,
    opts: SOMOptions,
) -> NifResult<Term<'a>> {
    let som = SOM::create(
        length,
        breadth,
        inputs,
        randomize,
        opts.learning_rate,
        opts.sigma,
        opts.decay_fn,
        opts.neighbourhood_fn,
        opts.classes,
        false,
    );
    Ok((ok(), ResourceArc::new(SomResource::from(som))).encode(env))
}

#[rustler::nif]
fn winner<'a>(env: rustler::Env<'a>, som: Rsc, sample: Vec<f64>) -> NifResult<Term<'a>> {
    return Ok(som.write().winner(Array1::from(sample)).encode(env));
}

#[rustler::nif]
fn train_random<'a>(
    env: rustler::Env<'a>,
    som: Rsc,
    data: Vec<Vec<f64>>,
    iterations: u32,
) -> NifResult<Term<'a>> {
    let inner_shape = data[0].len();
    let shape = (data.len(), inner_shape);
    let flat: Vec<f64> = data.iter().flatten().cloned().collect();
    som.write()
        .train_random(Array2::from_shape_vec(shape, flat).unwrap(), iterations)
        .encode(env);
    return Ok(ok().encode(env));
}

#[rustler::nif]
fn train_random_supervised<'a>(
    env: rustler::Env<'a>,
    som: Rsc,
    data: Vec<Vec<f64>>,
    class_data: Vec<String>,
    iterations: u32,
) -> NifResult<Term<'a>> {
    let inner_shape = data[0].len();
    let shape = (data.len(), inner_shape);
    let flat: Vec<f64> = data.iter().flatten().cloned().collect();
    som.write()
        .train_random_supervised(
            Array2::from_shape_vec(shape, flat).unwrap(),
            Array1::from(class_data),
            iterations,
        )
        .encode(env);
    return Ok(ok().encode(env));
}

#[rustler::nif]
fn train_random_hybrid<'a>(
    env: rustler::Env<'a>,
    som: Rsc,
    data: Vec<Vec<f64>>,
    class_data: Vec<String>,
    iterations: u32,
) -> NifResult<Term<'a>> {
    let inner_shape = data[0].len();
    let shape = (data.len(), inner_shape);
    let flat: Vec<f64> = data.iter().flatten().cloned().collect();
    som.write()
        .train_random_hybrid(
            Array2::from_shape_vec(shape, flat).unwrap(),
            Array1::from(class_data),
            iterations,
        )
        .encode(env);
    return Ok(ok().encode(env));
}

#[rustler::nif]
fn train_batch<'a>(
    env: rustler::Env<'a>,
    som: Rsc,
    data: Vec<Vec<f64>>,
    iterations: u32,
) -> NifResult<Term<'a>> {
    let inner_shape = data[0].len();
    let shape = (data.len(), inner_shape);
    let flat: Vec<f64> = data.iter().flatten().cloned().collect();
    som.write()
        .train_batch(Array2::from_shape_vec(shape, flat).unwrap(), iterations)
        .encode(env);
    return Ok(ok().encode(env));
}

#[rustler::nif]
fn lookup_tag<'a>(env: rustler::Env<'a>, som: Rsc, x: usize, y: usize) -> NifResult<Term<'a>> {
    //return Ok((ok(), Atom::from_str(env, &som.read().data.tag_map[[x, y]])?).encode(env))
    return Ok(ok().encode(env));
}

////////////////////////////////////////////////////////////////////////////
// Init                                                                   //
////////////////////////////////////////////////////////////////////////////

rustler::init!(
    "som",
    [
        new,
        winner,
        train_random,
        train_random_supervised,
        train_random_hybrid,
        train_batch,
        lookup_tag
    ],
    load = on_load
);

fn on_load<'a>(env: rustler::Env<'a>, _term: rustler::Term<'a>) -> bool {
    rustler::resource!(SomResource, env);
    true
}
