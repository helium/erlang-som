-module(iris_SUITE).

-include_lib("common_test/include/ct.hrl").
-include_lib("eunit/include/eunit.hrl").

-export([
         init_per_suite/1,
         end_per_suite/1,
         all/0
        ]).

-export([train_random/1, train_random_supervised/1]).

all() -> [
          train_random,
          train_random_supervised
         ].

init_per_suite(Config) ->
    {ok, IoDevice} = file:open("../../../../test/iris.csv", [read]),
    ct:pal("iodevice ~p", [IoDevice]),
    DisplayRow = fun({newline, [A, B, C, D, Class]}, Acc) ->
                         [{[list_to_float(A), list_to_float(B), list_to_float(C), list_to_float(D)], list_to_binary(Class)} | Acc];
                    (_, Acc) ->
                         Acc
                 end,
    {ok, ProcessedRows} = ecsv:process_csv_file_with(IoDevice, DisplayRow, []),
    ct:pal("processed ~p", [ProcessedRows]),
    [{rows, ProcessedRows}|Config].

end_per_suite(_Config) ->
    ok.

train_random(Config) ->
    Samples = ?config(rows, Config),
    {ok, SOM} = som:new(10, 10, 4, false),
    som:train_random(SOM, element(1, lists:unzip(Samples)), 1000),

    ?assert(false).

train_random_supervised(Config) ->
    Samples = ?config(rows, Config),
    {ok, SOM} = som:new(10, 10, 4, false, #{classes => #{<<"setosa">> => 0.0, <<"verisicolor">> => 0.0, <<"virginica">> => 0.0}}),
    %% use 60% of samples as training data
    {Supervised, Unsupervised} = lists:partition(fun(_) -> rand:uniform(100) < 60 end, Samples),
    {SupervisedSamples, SupervisedClasses} = lists:unzip(Supervised),
    som:train_random_supervised(SOM, SupervisedSamples, SupervisedClasses, 1000),
    [ ct:pal("~p ~p ~p", [Sample, Class, som:winner_vals(SOM, Sample)]) || {Sample, Class} <- Unsupervised ],
    ?assert(false).
