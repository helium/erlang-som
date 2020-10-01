-module(som).

-export([
    new/4
]).

-on_load(load/0).

-opaque som() :: reference().

-export_type([som/0]).

%% @doc Create a new self organizing map (SOM).
new(_Length, _Breadth, _Inputs, _Randomize) ->
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
