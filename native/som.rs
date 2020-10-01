use rusticsom::SOM;
use std::sync::RwLock;

////////////////////////////////////////////////////////////////////////////
// Resource                                                               //
////////////////////////////////////////////////////////////////////////////

#[repr(transparent)]
struct SomResource(RwLock<SOM>);

type Rsc = rustler::ResourceArc<SomResource>;

////////////////////////////////////////////////////////////////////////////
// NIFs                                                                   //
////////////////////////////////////////////////////////////////////////////

#[rustler::nif]
fn new(
    env: rustler::Env<'_>,
    length: usize,
    breadth: usize,
    inputs: usize,
    randomize: bool,
) -> Rsc {
    let _env = env;
    rustler::ResourceArc::new(SomResource(RwLock::new(SOM::create(
        length, breadth, inputs, randomize, None, None, None, None,
    ))))
}

////////////////////////////////////////////////////////////////////////////
// Init                                                                   //
////////////////////////////////////////////////////////////////////////////

rustler::init!("som", [new], load = on_load);

fn on_load<'a>(env: rustler::Env<'a>, _term: rustler::Term<'a>) -> bool {
    rustler::resource!(SomResource, env);
    true
}
