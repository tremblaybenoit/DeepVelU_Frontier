python deepvel_generate_input.py -output1 input1_tmp.fits -output2 input2_tmp.fits -directory input -nb_frames 31 -first_frame 1260
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end
mkdir output_simulation_series
python deepvel.py -i1 input1_tmp.fits -i2 input2_tmp.fits -o output_simulation_series/deepvel_output_simulation_series_00-31.fits -bx1 0 -bx2 11 -by1 0 -by2 11 &
PID=$! #catch the last PID, here from command1
wait $PID #wait for command1, in background, to end