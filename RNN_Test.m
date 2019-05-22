function [output ]= RNN_Test(net,TEST_INPUT)


N_Input = length(TEST_INPUT);


Applied_lambda = zeros(N_Input,1);             
Applied_LAMBDA = zeros(N_Input,1);


   for j=1:N_Input
      if (TEST_INPUT(j) >= 0)
         Applied_LAMBDA(j) = TEST_INPUT(j);
      elseif(TEST_INPUT(j) <0)
         Applied_lambda(j)= -TEST_INPUT(j);
      end
   end
 
   
%##### Preparing LAMBDA #################
fill = zeros(net.opts.N_Total - N_Input,1);
LAMBDA=[Applied_LAMBDA; fill];

%##### Preparing lambda #################
%Applied_lambda=zeros(N_Test_Patterns,N_Input);
fill = zeros(net.opts.N_Total - N_Input,1);
lambda=[Applied_lambda; fill];




Lamb.lambda = lambda';
Lamb.LAMBDA = LAMBDA';

[q,output_layer_start,output_layer_end,D] = Calc_Rate_and_Output( net,Lamb );
  
   output=[];
   for i=output_layer_start:output_layer_end
      output = [output q(i)];
   end
   
    

end


