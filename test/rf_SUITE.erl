-module(rf_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("eunit/include/eunit.hrl").

-export([
         init_per_suite/1,
         end_per_suite/1,
         all/0
        ]).

-export([train_random_supervised/1]).

all() -> [
          train_random_supervised
         ].

init_per_suite(Config) ->
    {ok, IoDevice} = file:open("../../../../test/rf.csv", [read]),
    ct:pal("iodevice ~p", [IoDevice]),
    DisplayRow = fun({newline, ["pos"|_]}, Acc) ->
                         %% ignore header
                         Acc;
                   ({newline, [_Pos, _Challengee, _Witness, Signal, SNR, FSPL, Class]}, Acc) ->
                         [{[((to_num(Signal) - (-145))/(145)), ((to_num(SNR) - (-18))/(18 - (-18))), ((to_num(FSPL) - (-145))/(145))], list_to_binary(Class)} | Acc];
                    (_, Acc) ->
                         Acc
                 end,
    {ok, ProcessedRows} = ecsv:process_csv_file_with(IoDevice, DisplayRow, []),
    ct:pal("processed ~p", [ProcessedRows]),
    [{rows, ProcessedRows}|Config].

end_per_suite(_Config) ->
    ok.

train_random_supervised(Config) ->
    Samples = ?config(rows, Config),
    {ok, SOM} = som:new(10, 10, 3, false, #{classes => #{<<"1">> => 1.7, <<"0">> => 0.6}, custom_weighting => false}),
    %% use 60% of samples as training data
    {Supervised, Unsupervised} = lists:partition(fun(_) -> rand:uniform(100) < 90 end, shuffle(Samples)),
    {SupervisedSamples, SupervisedClasses} = lists:unzip(Supervised),
    som:train_random_supervised(SOM, SupervisedSamples, SupervisedClasses, 2000),
    Matched = lists:foldl(fun({Sample, Class}, Acc) ->
                                  case som:winner_vals(SOM, Sample) of
                                      {_, Class} ->
                                          Acc + 1;
                                      {_, <<"0">>} ->
                                          %% false negative, leave this uncommented to make that not count as a fail
                                          Acc + 1;
                                      _ ->
                                          ct:pal("mismatch ~p ~p ~p", [Sample, Class, som:winner_vals(SOM, Sample)]),
                                          Acc
                                  end
                          end, 0, Unsupervised),
    ct:pal("matched ~p/~p => ~p%", [Matched, length(Unsupervised), Matched / length(Unsupervised) * 100]),
    ?assert(false).

to_num(String) ->
    try list_to_float(String) of
        Res -> Res
    catch _:_ ->
              list_to_integer(String) * 1.0
    end.

shuffle(List) ->
    [X || {_,X} <- lists:sort([{rand:uniform(), N} || N <- List])].
