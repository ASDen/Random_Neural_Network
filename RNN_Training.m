function [ net, err ] = RNN_Training( net,TRAIN_INPUT,TARGET )

N_Total =0;
for i=1:length(net.layers)
    N_Total = N_Total  + net.layers{1,i}.Number_of_Input_Neurons;
end

net.opts.N_Total  =N_Total ;


net.wplus = zeros(net.opts.N_Total,net.opts.N_Total);
net.wminus = zeros(net.opts.N_Total,net.opts.N_Total);

% Initializing the weights
%Input --->Hidden weights

start=0;
for i=1:length(net.layers)-1
    layer1_start =start+1;
    layer1_end = layer1_start+net.layers{1,i}.Number_of_Input_Neurons-1;
    layer2_start =layer1_end+1;
    layer2_end = layer2_start+net.layers{1,i+1}.Number_of_Input_Neurons-1;
    
    start = layer1_end;
    
    for ii = layer1_start:layer1_end
        jj = layer2_start:layer2_end;
        jjj=layer2_end-layer2_start+1;
            net.wplus(ii,jj)  = net.opts.RAND_RANGE*rand(1,jjj);
            net.wminus(ii,jj) = net.opts.RAND_RANGE*rand(1,jjj);
    end
    
end




N_Patterns = size(TRAIN_INPUT,1);          %Number of Patterns



%####### Setting the Excitatory and Inhibitatory Training Patterns ########
N_Train_Patterns = N_Patterns;
Applied_lambda = zeros(N_Train_Patterns,net.layers{1,1}.Number_of_Input_Neurons);

Applied_LAMBDA = zeros(N_Train_Patterns,net.layers{1,1}.Number_of_Input_Neurons);

Applied_y = TARGET;
%yyy = size(TRAIN_INPUT);
%N_Train_Patterns = yyy(1);
TRAIN_INPUT_Greater_Than_Zero=TRAIN_INPUT>0;
Applied_LAMBDA(TRAIN_INPUT_Greater_Than_Zero)= TRAIN_INPUT(TRAIN_INPUT_Greater_Than_Zero);
TRAIN_INPUT_Less_Than_Zero=TRAIN_INPUT<0;
Applied_lambda(TRAIN_INPUT_Less_Than_Zero)= -TRAIN_INPUT(TRAIN_INPUT_Less_Than_Zero);

%##### Preparing LAMBDA #################
fill = zeros(N_Train_Patterns , net.opts.N_Total - net.layers{1,1}.Number_of_Input_Neurons);
LAMBDA = [Applied_LAMBDA fill];

%##### Preparing lambda #################
%Applied_lambda=zeros(N_Train_Patterns,N_Input);
fill = zeros(N_Train_Patterns , net.opts.N_Total - net.layers{1,1}.Number_of_Input_Neurons);
lambda=[Applied_lambda fill];

%##### Preparing y #################
fill = zeros(N_Train_Patterns , net.opts.N_Total-net.layers{1,end}.Number_of_Input_Neurons);
y=[fill Applied_y];



iter=1;
elapsed_time=0;
while(iter <= net.opts.N_Iterations )
    
    t0 = clock;
    MSEaveg = 0.0;
    
    K1 = randperm(N_Patterns);
    
    for k = K1
        

	Lamb.lambda = lambda(k,:);
	Lamb.LAMBDA = LAMBDA(k,:);
        
        [q,output_layer_start,output_layer_end,D] = Calc_Rate_and_Output( net,Lamb );

   
        
        %Calc_MSE
        MSE=0.0;
        a = q(output_layer_start:output_layer_end) - y(k,output_layer_start:output_layer_end);
        MSE = MSE+sum(a.*a);
        
        
        MSEaveg = MSEaveg + MSE;
        %end Calc_MSE
        
        %updating the weights
        % Calculating W and (I-W)^-1 %
        W = zeros(net.opts.N_Total,net.opts.N_Total);
        for j = 1:net.opts.N_Total
             W(j,:) = (net.wplus(j,:) - (net.wminus(j,:).*q))./D;
        end
        
        inverse2 = inv(eye(net.opts.N_Total,net.opts.N_Total)-W);
        
        Winv = inverse2;
        % End Calc_Inv
        
        % Calc_Gamma
        % Calculating the weight updates
        start=0;
        for i=1:length(net.layers)-1
            layer1_start =start+1;
            layer1_end = layer1_start+net.layers{1,i}.Number_of_Input_Neurons-1;
            layer2_start =layer1_end+1;
            layer2_end = layer2_start+net.layers{1,i+1}.Number_of_Input_Neurons-1;
            
            start = layer1_end;
            
            for u = layer1_start:layer1_end
                for v = layer2_start:layer2_end
                    gammaplus=zeros(net.opts.N_Total,1);
                    gammaminus=zeros(net.opts.N_Total,1);
                    
                    if(u==v)
                        gammaminus(u) = -(1.0 + q(u))/D(u);
                    end
                    gammaplus(v) = 1.0/D(v);
                    gammaminus(v) = -q(v)/D(v);
                    gammaplus(u) = -1.0/D(u);
                    gammaminus(u) = -1.0/D(u);  
                            


                     vmplus = gammaplus' * Winv;
                        vmminus = gammaminus' * Winv;
                        i3 = output_layer_start:output_layer_end;
                        sum1=sum(vmplus(i3) .* (q(i3) - y(k,i3)))*q(u);
                        sum2=sum(vmminus(i3) .* (q(i3) - y(k,i3)))*q(u);
                        
                    
                    net.wplus(u,v) = net.wplus(u,v) - net.opts.Eta  * sum1;
                    net.wminus(u,v) = net.wminus(u,v) - net.opts.Eta  * sum2;
                    
                    
                    if (u == v)
                        net.wplus(u,v) = 0.0;
                        net.wminus(u,v) = 0.0;
                    end
                    if (net.wplus(u,v) < 0)
                        net.wplus(u,v) = 0.0;
                    end
                    if (net.wminus(u,v) < 0)
                        net.wminus(u,v) = 0.0;
                    end
                end
            end
        end
        
%         net.wminus(net.wminus<0)=0;
%         net.wplus(net.wplus<0)=0;
        
        
    end %k
    
    MSEaveg = MSEaveg/N_Patterns;
    [iter MSEaveg]
    t1 = etime(clock,t0);
    elapsed_time = elapsed_time + t1;
    err(iter) = MSEaveg;
    
    
    if (MSEaveg <= net.opts.Mse_Threshold)
        last_iter = iter;
        last_elapsed_time = elapsed_time;
        break;
    end
    
    
    last_iter = iter;
    last_elapsed_time = elapsed_time;
    iter = iter + 1;
    
end %while

end