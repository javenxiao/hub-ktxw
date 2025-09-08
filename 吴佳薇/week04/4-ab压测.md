#10并发总耗时如下：
(waimai_bert) PS F:\ai\Miniconda\Miniconda3\envs\waimai_bert\self_code\test\ApacheBench\Apache24\bin> ab -n 10 -c 10 http://127.0.0.1:8000/
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /
Document Length:        22 bytes

Concurrency Level:      10
Time taken for tests:   0.008 seconds
Complete requests:      10
Failed requests:        0
Non-2xx responses:      10
Total transferred:      1540 bytes
HTML transferred:       220 bytes
Requests per second:    1314.23 [#/sec] (mean)
Time per request:       7.609 [ms] (mean)
Time per request:       0.761 [ms] (mean, across all concurrent requests)
Transfer rate:          197.65 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.3      0       1
Processing:     2    3   1.1      4       4
Waiting:        1    2   1.2      2       4
Total:          2    3   1.1      4       4

Percentage of the requests served within a certain time (ms)
  50%      4
  66%      4
  75%      4
  80%      4
  90%      4
  95%      4
  98%      4
  99%      4
 100%      4 (longest request)





#5并发总耗时
(waimai_bert) PS F:\ai\Miniconda\Miniconda3\envs\waimai_bert\self_code\test\ApacheBench\Apache24\bin> ab -n 5 -c 5 http://127.0.0.1:8000/
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /
Document Length:        22 bytes

Concurrency Level:      5
Time taken for tests:   0.005 seconds
Complete requests:      5
Failed requests:        0
Non-2xx responses:      5
Total transferred:      770 bytes
HTML transferred:       110 bytes
Requests per second:    996.21 [#/sec] (mean)
Time per request:       5.019 [ms] (mean)
Time per request:       1.004 [ms] (mean, across all concurrent requests)
Transfer rate:          149.82 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.5      0       1
Processing:     1    1   0.4      1       2
Waiting:        1    1   0.1      1       1
Total:          1    2   0.7      2       3

Percentage of the requests served within a certain time (ms)
  50%      1
  66%      2
  75%      2
  80%      3
  90%      3
  95%      3
  98%      3
  99%      3
 100%      3 (longest request)
(waimai_bert) PS F:\ai\Miniconda\Miniconda3\envs\waimai_bert\self_code\test\ApacheBench\Apache24\bin>



(waimai_bert) PS F:\ai\Miniconda\Miniconda3\envs\waimai_bert\self_code\test\ApacheBench\Apache24\bin> ab -n 1 -c 1 http://127.0.0.1:8000/
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /
Document Length:        22 bytes

Concurrency Level:      1
Time taken for tests:   0.001 seconds
Complete requests:      1
Failed requests:        0
Non-2xx responses:      1
Total transferred:      154 bytes
HTML transferred:       22 bytes
Requests per second:    991.08 [#/sec] (mean)
Time per request:       1.009 [ms] (mean)
Time per request:       1.009 [ms] (mean, across all concurrent requests)
Transfer rate:          149.05 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:     1    1   0.0      1       1
Waiting:        1    1   0.0      1       1
Total:          1    1   0.0      1       1



