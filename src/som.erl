-module(som).

-export([
    new/5,
    winner/2,
    train_random/3,
    train_random_supervised/4,
    train_random_hybrid/4,
    train_batch/3
]).

-on_load(load/0).

-opaque som() :: reference().

-export_type([som/0]).

%% @doc Create a new self organizing map (SOM).
new(_Length, _Breadth, _Inputs, _Randomize, _Options) ->
    not_loaded(?LINE).

winner(_Som, _Sample) ->
    not_loaded(?LINE).

train_random(_Som, _Data, _Iterations) ->
    not_loaded(?LINE).

train_random_supervised(_Som, _Data, _Iterations, _Classes) ->
    not_loaded(?LINE).

train_random_hybrid(_Som, _Data, _Iterations, _Classes) ->
    not_loaded(?LINE).

train_batch(_Som, _Data, _Iterations) ->
    not_loaded(?LINE).

load() ->
    erlang:load_nif(filename:join(priv(), "libsom"), none).

not_loaded(Line) ->
    erlang:nif_error({error, {not_loaded, [{module, ?MODULE}, {line, Line}]}}).

priv() ->
    case code:priv_dir(?MODULE) of
        {error, _} ->
            EbinDir = filename:dirname(code:which(?MODULE)),
            AppPath = filename:dirname(EbinDir),
            filename:join(AppPath, "priv");
        Path ->
            Path
    end.
