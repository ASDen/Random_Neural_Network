net.layers = {} ;
net.layers{end+1} = struct('name','input', 'Number_of_Input_Neurons',4);
net.layers{end+1} = struct('name','hidden', 'Number_of_Input_Neurons',5);
net.layers{end+1} = struct('name','hidden', 'Number_of_Input_Neurons',5);
net.layers{end+1} = struct('name','output', 'Number_of_Input_Neurons',3);


net.opts.Mse_Threshold      = 0.000002;            %Required Stop Mean Square Error value
net.opts.Eta                = 0.05;                 %Learinig Rate
net.opts.N_Iterations       = 200;                 %Maximum Number of Iterations
net.opts.R_Out              = .1;                 %Firing Rate of the Output Neurons
net.opts.RAND_RANGE         = .1;                  %Random Range of the Weights 
net.opts.FIX_RIN            = 0;                   %Flag for Fixing the Input Firing Rate 
net.opts.R_IN               = 1;                   %Value of the Input Firing Rate(if Fixed)
