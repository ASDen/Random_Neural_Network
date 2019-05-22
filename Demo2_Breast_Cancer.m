clear all
Data_Folder_Name = 'Demo2_Breast_Cancer';

Parameters_File_Name = fullfile(Data_Folder_Name, 'Parameters.m');
run(Parameters_File_Name);

Train_File_Name = fullfile(Data_Folder_Name, 'Train.mat');
Train1 = load(Train_File_Name);

TRAIN_INPUT=Train1.INPUT;
TARGET=Train1.TARGET;
rng(0); 
[ net, err ] = RNN_Training( net,TRAIN_INPUT,TARGET );

Test_File_Name = fullfile(Data_Folder_Name, 'Test.mat');

Test1 = load(Test_File_Name);
TEST_INPUT = Test1.INPUT;
TEST_LABEL = Test1.LABEL;

No_of_classes = size(TEST_LABEL,2);
N_Test_Patterns = size(TEST_INPUT,1);
Conf_Matrix = zeros(No_of_classes,No_of_classes);
total_error = 0;
error_k1 = zeros(N_Test_Patterns,1);
for i=1:N_Test_Patterns

    output= RNN_Test(net,TEST_INPUT(i,:));

    true_class = find(TEST_LABEL(i,:));
       
   [max1 out_class] = max(output);
   
   if(true_class ~= out_class)
       total_error=total_error+1;
       error_k1(i)=1;
   end

   Conf_Matrix(true_class,out_class) = Conf_Matrix(true_class,out_class)+1;
end

Percent_error = total_error*100/N_Test_Patterns
Percent_True = (N_Test_Patterns-total_error)*100/N_Test_Patterns

Conf_Matrix