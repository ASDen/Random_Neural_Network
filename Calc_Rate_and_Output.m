function [q,output_layer_start,output_layer_end,D] = Calc_Rate_and_Output( net,Lamb )

        % Calc_Rate
        r=zeros(net.opts.N_Total,1);
        
        start=0;
        for i=1:length(net.layers)-1
            layer1_start =start+1;
            layer1_end = layer1_start+net.layers{1,i}.Number_of_Input_Neurons-1;
            layer2_start =layer1_end+1;
            layer2_end = layer2_start+net.layers{1,i+1}.Number_of_Input_Neurons-1;
            
            start = layer1_end;
            iii1 = layer1_start:layer1_end;
            iii2=layer2_start:layer2_end;
            r(iii1)=sum(net.wplus(iii1,iii2)')+sum(net.wminus(iii1,iii2)');
            
        end
        
        
        if(net.opts.FIX_RIN ==1)
            r(1:net.layers{1,1}.Number_of_Input_Neurons) = net.opts.R_IN;
        end
        output_layer_start = layer2_start;
        output_layer_end = layer2_end;
        
        r(output_layer_start:output_layer_end)=net.opts.R_Out;
        
        % end Calc_Rate %


        %calculation of the Actual Output
        N = Lamb.LAMBDA(1:net.layers{1,1}.Number_of_Input_Neurons);
        D = Lamb.lambda(1:net.layers{1,1}.Number_of_Input_Neurons) + r(1:net.layers{1,1}.Number_of_Input_Neurons)';
        q=N./D;
       q(q>1.0) = 1;
        
        start=0;
        for i=1:length(net.layers)-1
            layer1_start =start+1;
            layer1_end = layer1_start+net.layers{1,i}.Number_of_Input_Neurons-1;
            layer2_start =layer1_end+1;
            layer2_end = layer2_start+net.layers{1,i+1}.Number_of_Input_Neurons-1;
            
            start = layer1_end;
            
            N(layer2_start:layer2_end) = 0.0;
            D(layer2_start:layer2_end) = 0.0;
            for i1 = layer2_start:layer2_end
                N(i1) = N(i1) + sum(q(layer1_start:layer1_end) .* net.wplus(layer1_start:layer1_end,i1)');
                D(i1) = D(i1) + sum(q(layer1_start:layer1_end) .* net.wminus(layer1_start:layer1_end,i1)');
                N(i1) = N(i1)+Lamb.LAMBDA(i1);
                D(i1) = D(i1) + r(i1)+Lamb.lambda(i1);
                q(i1) = N(i1)/D(i1);
            end
            
        end
        
     % end Calc_Output

        output_layer_start = layer2_start;
        output_layer_end = layer2_end;

