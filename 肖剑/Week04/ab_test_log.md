(py312) PS E:\tools\01-intent-classify\test> ab -n 100 -c 1 -p data.json -T 'application/json' -H 'accept: application/json' '<http://127.0.0.1:8000/v1/text-cls/bert>'
This is ApacheBench, Version 2.3 <`$Revision: 1843412 $`>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, <http://www.zeustech.net/>
Licensed to The Apache Software Foundation, <http://www.apache.org/>

Benchmarking 127.0.0.1 (be patient).....done

Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        167 bytes

Concurrency Level:      1
Time taken for tests:   4.147 seconds
Complete requests:      100
Failed requests:        12
(Connect: 0, Receive: 0, Length: 12, Exceptions: 0)
Total transferred:      29288 bytes
Total body sent:        27500
HTML transferred:       16688 bytes
Requests per second:    24.12 \[#/sec] (mean)
Time per request:       41.467 \[ms] (mean)
Time per request:       41.467 \[ms] (mean, across all concurrent requests)
Transfer rate:          6.90 \[Kbytes/sec] received
6.48 kb/s sent
13.37 kb/s total

Connection Times (ms)
min  mean\[+/-sd] median   max
Connect:        0    0   0.4      0       1
Processing:    37   41   2.3     41      50
Waiting:       37   41   2.3     40      50

Percentage of the requests served within a certain time (ms)
50%     41
66%     42
75%     42
80%     43
90%     45
95%     46
98%     49
99%     50
100%     50 (longest request)
(py312) PS E:\tools\01-intent-classify\test> ab -n 100 -c 5 -p data.json -T 'application/json' -H 'accept: application/json' '<http://127.0.0.1:8000/v1/text-cls/bert>'
This is ApacheBench, Version 2.3 <`$Revision: 1843412 $`>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, <http://www.zeustech.net/>
Licensed to The Apache Software Foundation, <http://www.apache.org/>

Benchmarking 127.0.0.1 (be patient).....done

Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        167 bytes

Concurrency Level:      5
Time taken for tests:   3.233 seconds
Complete requests:      100
Failed requests:        16
(Connect: 0, Receive: 0, Length: 16, Exceptions: 0)
Total transferred:      29284 bytes
Total body sent:        27500
HTML transferred:       16684 bytes
Requests per second:    30.93 \[#/sec] (mean)
Time per request:       161.643 \[ms] (mean)
Time per request:       32.329 \[ms] (mean, across all concurrent requests)
Transfer rate:          8.85 \[Kbytes/sec] received
8.31 kb/s sent
17.15 kb/s total

Connection Times (ms)
min  mean\[+/-sd] median   max
Connect:        0    0   0.5      0       1
Processing:    46  158  16.3    163     182
Waiting:       45  157  16.4    162     181

Percentage of the requests served within a certain time (ms)
50%    163
66%    165
75%    167
80%    168
90%    172
95%    176
98%    178
99%    182
100%    182 (longest request)
(py312) PS E:\tools\01-intent-classify\test> ab -n 100 -c 100 -p data.json -T 'application/json' -H 'accept: application/json' '<http://127.0.0.1:8000/v1/text-cls/bert>'
This is ApacheBench, Version 2.3 <`$Revision: 1843412 $`>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, <http://www.zeustech.net/>
Licensed to The Apache Software Foundation, <http://www.apache.org/>

Benchmarking 127.0.0.1 (be patient).....done

Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        167 bytes

Concurrency Level:      100
Time taken for tests:   3.372 seconds
Complete requests:      100
Failed requests:        11
(Connect: 0, Receive: 0, Length: 11, Exceptions: 0)
Total transferred:      29288 bytes
Total body sent:        27500
HTML transferred:       16688 bytes
Requests per second:    29.65 \[#/sec] (mean)
Time per request:       3372.402 \[ms] (mean)
Time per request:       33.724 \[ms] (mean, across all concurrent requests)
Transfer rate:          8.48 \[Kbytes/sec] received
7.96 kb/s sent
16.44 kb/s total

Connection Times (ms)
min  mean\[+/-sd] median   max
Connect:        0    0   0.3      0       1
Processing:    60 3079 352.4   3180    3313
Waiting:       45 3016 350.6   3067    3313
Total:         60 3079 352.4   3180    3313

Percentage of the requests served within a certain time (ms)
50%   3180
66%   3238
75%   3239
80%   3239
90%   3292
95%   3309
98%   3313
99%   3313
100%   3313 (longest request)
(py312) PS E:\tools\01-intent-classify\test>
