set terminal postscript eps enhanced color font 'Helvetica,16'
set output 'objective.eps' 
#set terminal x11 enhanced font "Helvetica,14" 0

set xlabel "Number of UEs"
set ylabel "Sum of QoE values"
set title "Objective function value"

set key top left 
plot \
  "./results.dat" u 1:2:3:4 t "Optimal" with errorlines pt 1 ps 2, \
  "./results.dat" u 1:5:6:7 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:8:9:10 t "Baseline" with errorlines pt 3 ps 2

#pause -1

#set terminal postscript eps enhanced color font 'Helvetica,14'
#set terminal x11 enhanced font "Helvetica,14" 1
set output 'time.eps' 
set key top left 
set xlabel "Number of UEs"
set ylabel "Time (s)"
set title "Execution time"
plot \
  "./results.dat" u 1:11:12:13 t "Optimal" with errorlines pt 1 ps 2, \
  "./results.dat" u 1:14:15:16 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:17:18:19 t "Baseline" with errorlines pt 3 ps 2

#pause -1

#set terminal postscript eps enhanced color font 'Helvetica,14'
#set terminal x11 enhanced font "Helvetica,14" 2
set output 'optimality.eps' 
set key bottom right
set xlabel "Number of UEs"
set ylabel "Ratio of the optimal solution value"
set title "Optimality"
set yrange [0:1]
plot \
  "./results.dat" u 1:20:21:22 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:23:24:25 t "Baseline" with errorlines pt 3 ps 2

#pause -1
#############################
set output "threads.eps"
set xlabel "Number of UEs"
set ylabel "Execution time"
set yrange [0:2]
set xrange [10:2000]
set title "Execution time vs. number of available CPU cores"
plot \
  "./threads.dat" u 1:2:3:4 t "1 CPU" w errorlines pt 1 ps 2,\
  "./threads.dat" u 1:5:6:7 t "2 CPU" w errorlines pt 2 ps 2,\
  "./threads.dat" u 1:8:9:10 t "3 CPU" w errorlines pt 3 ps 2,\
  "./threads.dat" u 1:11:12:13 t "4 CPU" w errorlines pt 4 ps 2
##############################

set output "mos.eps"
set key top right
set xlabel "Number of UEs"
set ylabel "MOS"
set title "Average QoE (including UEs not assigned video)"
set yrange [0:5]

plot \
  "./results.dat" u 1:26:27:28 t "Optimal" with errorlines pt 1 ps 2, \
  "./results.dat" u 1:29:30:31 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:32:33:34 t "Baseline" with errorlines pt 3 ps 2

set output "mos_served.eps"
set xlabel "Number of UEs"
set ylabel "MOS"
set title "Average QoE (only including UEs assigned video)"
set yrange [0:5]

plot \
  "./results.dat" u 1:35:36:37 t "Optimal" with errorlines pt 1 ps 2, \
  "./results.dat" u 1:38:39:40 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:41:42:43 t "Baseline" with errorlines pt 3 ps 2

set output "ratio.eps"
set key top right
set xlabel "Number of UEs"
set ylabel "Ratio"
set title "Ratio of UEs served (receiving at least the lowest quality video)"
set yrange [0:1]

plot \
  "./results.dat" u 1:44:45:46 t "Optimal" with errorlines pt 1 ps 2, \
  "./results.dat" u 1:47:48:49 t "GA" with errorlines pt 2 ps 2, \
  "./results.dat" u 1:50:51:52 t "Baseline" with errorlines pt 3 ps 2

