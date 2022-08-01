close all
clear all

file_path = './Test Runs - NG';
file_path2 = './Test Runs - NG/Mismatched Targets - NG';
file_path3 = './Test Runs - NG/Mismatched Targets - NG 2';
file_path4 = './Test Runs - NG/PFA Characterization';
file_path5 = './Test Runs - NG/PFA Characterization/Correlation';
test_num = 93;
angle_test_num = 40;
num_runs = 26;


% load('./Test Runs/Test-1-PFA_3-N_25-K_75-SNIR_10_35-Sim_Num_1e5-Gaussian.mat')


FA_ace_tests = zeros(test_num,num_runs);
FA_amf_tests = zeros(test_num,num_runs);
FA_glrt_tests = zeros(test_num,num_runs);
tally_ace_tests = zeros(test_num,num_runs);
tally_amf_tests = zeros(test_num,num_runs);
tally_glrt_tests = zeros(test_num,num_runs);
cos2theta_tests = zeros(angle_test_num,1);


%Load data and generate plots for performance comparison in Gaussian case
for i=1:test_num
    if i<17
        file_pattern = fullfile(file_path,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path,file_name);
        load(full_file_name)
    elseif i<37
        file_pattern = fullfile(file_path2,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path2,file_name);
        load(full_file_name)
        cos2theta_tests(i-16) = cos2theta;
    elseif i<57
        file_pattern = fullfile(file_path3,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path3,file_name);
        load(full_file_name)
        cos2theta_tests(i-16) = cos2theta;
    elseif i<74
        file_pattern = fullfile(file_path4,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path4,file_name);
        load(full_file_name)
    elseif i<94
        file_pattern = fullfile(file_path5,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path5,file_name);
        load(full_file_name)
    end
    
    
    FA_ace_tests(i,:) = FA_ace;
    FA_amf_tests(i,:) = FA_amf;
    FA_glrt_tests(i,:) = FA_glrt;
    tally_ace_tests(i,:) = tally_ace;
    tally_amf_tests(i,:) = tally_amf;
    tally_glrt_tests(i,:) = tally_glrt;
    
    %Test over different values of the clutter scale parameter b
    if i<3
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N, P_fa = 1e-3, a = 0.5, b = 1,4
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Scale Parameter: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = 0.5, b = ',num2str(4-3*(i-1))))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Scale Parameter - b_',num2str(4-3*(i-1)),' - K'), 'pdf') %Save figure
    
    %Test over different values of the clutter shape parameter a    
    elseif i<5
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N, P_fa = 1e-3, a = 0.5,1.5,4.5, b = 1
        if i==3
            figure
            plot(SNIR,tally_ace_tests(i-1,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-1,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-1,:)/1e5)
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Shape Parameter - a_0_5 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Shape Parameter: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = ',num2str(0.5+(3*(i-3)+1)), 'b = 1'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Shape Parameter - a_',num2str((3*(i-3)+1)),'_5 - K'), 'pdf') %Save figure
        
    %Test over different values of N (signal length)
    elseif i<7
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5,15,25, K = 5*N, P_fa = 1e-3, a = 4.5, b = 1
        if i==5
            figure
            plot(SNIR,tally_ace_tests(i-1,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-1,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-1,:)/1e5)
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%             saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - N_5 - a_4_5 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Sample Support: P_{d} vs SNIR')
%         subtitle(strcat('N = ',num2str(15+10*(i-5)),', K = 5*N, a = 4.5, b = 1'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%         saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - N_',num2str(15+10*(i-5)),' - a_4_5 - K'), 'pdf') %Save figure
        
    
    %Test over different values of N (signal length)
    elseif i<9
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5,15,25, K = 5*N, P_fa = 1e-3, a = 0.5, b = 1
        if i==7
            figure
            plot(SNIR,tally_ace_tests(i-5,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-5,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-5,:)/1e5)
            plot(SNIR,tally_ace_tests(i-3,:)/1e5,'--')
            plot(SNIR,tally_amf_tests(i-3,:)/1e5,'--')
            plot(SNIR,tally_glrt_tests(i-3,:)/1e5,'--')
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold')
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - N_5 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
        plot(SNIR,tally_ace_tests(i-(2-mod(i,2)),:)/1e5,'--')
        plot(SNIR,tally_amf_tests(i-(2-mod(i,2)),:)/1e5,'--')
        plot(SNIR,tally_glrt_tests(i-(2-mod(i,2)),:)/1e5,'--')
%         title('Impact of Sample Support: P_{d} vs SNIR')
%         subtitle(strcat('N = ',num2str(25-10*(i-7)),', K = 5*N, a = 0.5, b = 1'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold')
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - N_',num2str(25-10*(i-7)),' - K'), 'pdf') %Save figure
        
        
    %Test over different values of K (signal length)
    elseif i<11
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 3,4,5*N, P_fa = 1e-3, a = 0.5, b = 1
        if i==9
            figure
            plot(SNIR,tally_ace_tests(i-2,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-2,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-2,:)/1e5)
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%             saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - N_25 - K_5N - a_0_5 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Sample Support: P_{d} vs SNIR')
%         subtitle(strcat('N = 25, K = ',num2str(3+mod(i,2)),'*N, a = 0.5, b = 1'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%         saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - N_25 - K_',num2str(3+mod(i,2)),'N - a_0_5 - K'), 'pdf') %Save figure
        
    %Test over different values of K (signal length)
    elseif i<13
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 3,4,5*N, P_fa = 1e-3, a = 4.5, b = 1
        if i==11
            figure
            plot(SNIR,tally_ace_tests(i-4,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-4,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-4,:)/1e5)
            plot(SNIR,tally_ace_tests(i-5,:)/1e5,'--')
            plot(SNIR,tally_amf_tests(i-5,:)/1e5,'--')
            plot(SNIR,tally_glrt_tests(i-5,:)/1e5,'--')
            
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - N_25 - K_5N - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i-2,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i-2,:)/1e5)
        plot(SNIR,tally_glrt_tests(i-2,:)/1e5)
        plot(SNIR,tally_ace_tests(i,:)/1e5,'--')
        plot(SNIR,tally_amf_tests(i,:)/1e5,'--')
        plot(SNIR,tally_glrt_tests(i,:)/1e5,'--')        
%         title('Impact of Sample Support: P_{d} vs SNIR')
%         subtitle(strcat('N = 25, K = ',num2str(3+mod(i,2)),'*N, a = 4.5, b = 1'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold')
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - N_25 - K_',num2str(3+mod(i,2)),'N - K'), 'pdf') %Save figure
        
    
    %Test over different values of rho (one-lag correlation coefficient)
    elseif i<15
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N P_fa = 1e-3, a = 4.5, b = 1 with rho = 0.8,0.5,0.1 
        if i==13
            figure
            plot(SNIR,tally_ace_tests(i-9,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-9,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-9,:)/1e5)
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%             saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - rho_0_9 - a_4_5 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Correlation: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = 4.5, b = 1, \rho = ',num2str(0.5-0.4*(i-13))))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','northwest','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
%         saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - rho_0_',num2str(5-4*(i-13)),' - a_4_5 - K'), 'pdf') %Save figure
    
    
    %Test over different values of rho (one-lag correlation coefficient)
    elseif i<17
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N P_fa = 1e-3, a = 4.5, b = 1 with rho = 0.9,0.5,0.1 
        if i==15
            figure
            plot(SNIR,tally_ace_tests(i-13,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-13,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-13,:)/1e5)
            plot(SNIR,tally_ace_tests(i-11,:)/1e5,'--')
            plot(SNIR,tally_amf_tests(i-11,:)/1e5,'--')
            plot(SNIR,tally_glrt_tests(i-11,:)/1e5,'--')
%             title('Impact of Shape Parameter: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, a = 0.5,b = 1'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold')  
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - rho_0_9 - K'), 'pdf') %Save figure
        end
        
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
        plot(SNIR,tally_ace_tests(i-2,:)/1e5,'--')
        plot(SNIR,tally_amf_tests(i-2,:)/1e5,'--')
        plot(SNIR,tally_glrt_tests(i-2,:)/1e5,'--')
%         title('Impact of Correlation: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = 4.5, b = 1, \rho = ',num2str(0.5-0.4*(i-15))))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE, a = 0.5','AMF, a = 0.5','GLRT, a = 0.5','ACE, a = 4.5','AMF, a = 4.5','GLRT, a = 4.5','Location','northwest','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - rho_0_',num2str(5-4*(i-15)),' - K'), 'pdf') %Save figure
    
% %     Test over different values of target return misalignment
%     elseif i<37
%         %Plot the Probability of Detection vs the SNIR for each detector
%         %at N = 5, K = 5*N P_fa = 1e-3 at a range of signal mismatch angles
%         figure
%         plot(SNIR,tally_ace_tests(i,:)/1e5)
%         hold on
%         plot(SNIR,tally_amf_tests(i,:)/1e5)
%         plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = 4.5, cos^{2}(\theta) = ',num2str(real(cos2theta_tests(i-16)))))
%         xlabel('SNIR (dB)')
%         ylabel('P_{d}')
%         legend('ACE','AMF','GLRT','Location','southeast')
%     
%     elseif i<57
%         %Plot the Probability of Detection vs the SNIR for each detector
%         %at N = 5, K = 5*N P_fa = 1e-3 at a range of signal mismatch angles
%         figure
%         plot(SNIR,tally_ace_tests(i,:)/1e5)
%         hold on
%         plot(SNIR,tally_amf_tests(i,:)/1e5)
%         plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, a = 4.5, cos^{2}(\theta) = ',num2str(real(cos2theta_tests(i-16)))))
%         xlabel('SNIR (dB)')
%         ylabel('P_{d}')
%         legend('ACE','AMF','GLRT','Location','southeast')
    elseif (56<i)&&(i<74)
        a_tests(i-56,:) = a;
        if (i-56)==7
            figure
            histogram(abs(d),50,'Normalization','probability')
            xlabel('Clutter Magnitude','fontweight','bold')
            ylabel('Occurrence in Generated Data (%)','fontweight','bold')
            axis([0 15 0 0.2])
            box on
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Distribution - a_25 - NG', 'pdf') %Save figure
        elseif (i-56)==10
            figure
            histogram(abs(d),50,'Normalization','probability')
            xlabel('Clutter Magnitude','fontweight','bold')
            ylabel('Occurrence in Generated Data (%)','fontweight','bold')
            axis([0 15 0 0.2])
            box on
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Distribution - a_4_5 - NG', 'pdf') %Save figure
        elseif (i-56)==12
            figure
            histogram(abs(d),50,'Normalization','probability')
            xlabel('Clutter Magnitude','fontweight','bold')
            ylabel('Occurrence in Generated Data (%)','fontweight','bold')
            axis([0 15 0 0.2])
            box on
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Distribution - a_1_5 - NG', 'pdf') %Save figure
        elseif (i-56)==14
            figure
            histogram(abs(d),50,'Normalization','probability')
            xlabel('Clutter Magnitude','fontweight','bold')
            ylabel('Occurrence in Generated Data (%)','fontweight','bold')
            axis([0 15 0 0.2])
            box on
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Distribution - a_0_5 - NG', 'pdf') %Save figure
%         elseif (i-56)==16
%             figure
%             histogram(real(d)+imag(d),50,'Normalization','pdf')
        end
    elseif (i>73) && (i<94)
        rho_tests(i-73,:) = rho;    
    end
end


%% for each detector, plot the false alarm rate as a function of the one-lag correlation coefficient
figure
hold on
plot(rho_tests(1:10),log10(FA_ace_tests(74:83)/1e5))
plot(rho_tests(1:10),log10(FA_amf_tests(74:83)/1e5))
plot(rho_tests(1:10),log10(FA_glrt_tests(74:83)/1e5))
plot(rho_tests(11:20),log10(FA_ace_tests(84:93)/1e5),'--')
plot(rho_tests(11:20),log10(FA_amf_tests(84:93)/1e5),'--')
plot(rho_tests(11:20),log10(FA_glrt_tests(84:93)/1e5),'--')
axis([0 1 -4 0])
yl = yline(-3,'--','P_{fa} = 10^{-3}');
yl.LabelHorizontalAlignment = 'left';
xlabel('One-Lag Clutter Correlation Coefficient (\rho)','fontweight','bold')
ylabel('log_{10}(P_{fa})','fontweight','bold')
box on
legend('ACE - a = 4.5','AMF - a = 4.5','GLRT - a = 4.5','ACE - a = 0.5','AMF - a = 0.5','GLRT - a = 0.5','location','northwest','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation PFA - Wide - K', 'pdf') %Save figure


%% for each detector, plot the false alarm rate as a function of the clutter shape parameter
figure
hold on
plot(log10(a_tests),log10(FA_ace_tests(57:73,1)/1e5))
plot(log10(a_tests),log10(FA_amf_tests(57:73,1)/1e5))
plot(log10(a_tests),log10(FA_glrt_tests(57:73,1)/1e5))
yl = yline(-3,'--','P_{fa} = 10^{-3}');
yl.LabelHorizontalAlignment = 'left';
xl = xline(1.4,'--','a = 25');
xl.LabelOrientation = 'horizontal';
xl.LabelVerticalAlignment = 'middle';
xl2 = xline(0,'--','a = 1');
xl2.LabelOrientation = 'horizontal';
xl2.LabelVerticalAlignment = 'middle';
% axis([-2 3 0 0.01])
xlabel('Clutter Shape Parameter (log_{10}(a))','fontweight','bold')
ylabel('log_{10}(P_{fa})','fontweight','bold')
box on
legend('ACE','AMF','GLRT','location','northeast','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\PFA - Wide - NG', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR with respect to signal return misalignment with shape parameter a = 0.5.
figure
hold on
for i=1:20
    plot(SNIR,tally_glrt_tests(16+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('GLRT Detector, N = 5, K = 5*N, a = 0.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.48 0.7];
y = [0.6 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.39 0.6 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA - GLRT - NG', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_amf_tests(16+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('AMF Detector, N = 5, K = 5*N, a = 0.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.59 0.7];
y = [0.4 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.5 0.4 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA - AMF - NG', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_ace_tests(16+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('ACE Detector, N = 5, K = 5*N, a = 0.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.4 0.7];
y = [0.6 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.3 0.6 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA - ACE - NG', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR with respect to signal return misalignment with shape parameter a = 4.5.
figure
hold on
for i=1:20
    plot(SNIR,tally_glrt_tests(36+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('GLRT Detector, N = 5, K = 5*N, a = 4.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.38 0.7];
y = [0.6 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.275 0.6 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA 2 - GLRT - NG', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_amf_tests(36+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('AMF Detector, N = 5, K = 5*N, a = 4.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.38 0.7];
y = [0.5 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.275 0.5 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA 2 - AMF - NG', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_ace_tests(36+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('ACE Detector, N = 5, K = 5*N, a = 4.5, b = 1')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.5 0.7];
y = [0.51 0.115];
annotation('arrow',x,y)
% annotation('textbox',[0.395 0.5 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
annotation('textbox',[0.385 0.51 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA 2 - ACE - NG', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for the K-distribution with different scale parameters
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5, b = 4,1
figure
plot(SNIR,tally_ace_tests(1,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(2,:)/1e5)
% title('Impact of Distribution Scale Parameter: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N, a = 0.5, b = 1,4'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('b = 4','b = 1','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Scale Parameter - ACE - K'), 'pdf') %Save figure


%Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5, b = 4,1
figure
plot(SNIR,tally_amf_tests(1,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(2,:)/1e5)
% title('Impact of Distribution Scale Parameter: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N, a = 0.5, b = 1,4'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('b = 4','b = 1','Location','northwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Scale Parameter - AMF - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5, b = 4,1
figure
plot(SNIR,tally_glrt_tests(1,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(2,:)/1e5)
% title('Impact of Distribution Scale Parameter: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N, a = 0.5, b = 1,4'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('b = 4','b = 1','Location','northwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Scale Parameter - GLRT - K'), 'pdf') %Save figure

%% For each detector, plot the probability of detection vs SNIR for the K-distribution with different shape parameters
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5,1.5,4.5
figure
plot(SNIR,tally_ace_tests(2,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(3,:)/1e5)
plot(SNIR,tally_ace_tests(4,:)/1e5)
% title('Impact of Distribution Shape Parameter: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N, a = 0.5,1.5,4.5 b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('a = 0.5','a = 1.5','a = 4.5','Location','northwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Shape Parameter - ACE - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5,1.5,4.5
figure
plot(SNIR,tally_amf_tests(2,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(3,:)/1e5)
plot(SNIR,tally_amf_tests(4,:)/1e5)
% title('Impact of Distribution Shape Parameter: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N, a = 0.5,1.5,4.5 b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('a = 0.5','a = 1.5','a = 4.5','Location','northwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Shape Parameter - AMF - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 5,
%K = 5*N, P_fa = 1e-3, a = 0.5,1.5,4.5
figure
plot(SNIR,tally_glrt_tests(2,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(3,:)/1e5)
plot(SNIR,tally_glrt_tests(4,:)/1e5)
% title('Impact of Distribution Shape Parameter: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N, a = 0.5,1.5,4.5 b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('a = 0.5','a = 1.5','a = 4.5','Location','northwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Shape Parameter - GLRT - K'), 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of N 
%Plot the Probability of Detection vs the SNIR for ACE detector at 
%N = 5,15,25, K = 5*N, P_fa = 1e-3, a = 0.5,4.5, b = 1
figure
plot(SNIR,tally_ace_tests(2,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(8,:)/1e5)
plot(SNIR,tally_ace_tests(7,:)/1e5)
plot(SNIR,tally_ace_tests(4,:)/1e5,'--')
plot(SNIR,tally_ace_tests(5,:)/1e5,'--')
plot(SNIR,tally_ace_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5, a = 0.5','N = 15, a = 0.5','N = 25, a = 0.5','N = 5, a = 4.5','N = 15, a = 4.5','N = 25, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - ACE - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_amf_tests(2,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(8,:)/1e5)
plot(SNIR,tally_amf_tests(7,:)/1e5)
plot(SNIR,tally_amf_tests(4,:)/1e5,'--')
plot(SNIR,tally_amf_tests(5,:)/1e5,'--')
plot(SNIR,tally_amf_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5, a = 0.5','N = 15, a = 0.5','N = 25, a = 0.5','N = 5, a = 4.5','N = 15, a = 4.5','N = 25, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - AMF - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_glrt_tests(2,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(8,:)/1e5)
plot(SNIR,tally_glrt_tests(7,:)/1e5)
plot(SNIR,tally_glrt_tests(4,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(5,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5, a = 0.5','N = 15, a = 0.5','N = 25, a = 0.5','N = 5, a = 4.5','N = 15, a = 4.5','N = 25, a = 4.5','Location','northwest','fontweight','bold')     
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Size - GLRT - K'), 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of K at a given N 
%Plot the Probability of Detection vs the SNIR for ACE detector at 
%N = 5,15,25, K = 5*N, P_fa = 1e-3, a = 0.5,4.5, b = 1
figure
plot(SNIR,tally_ace_tests(10,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(9,:)/1e5)
plot(SNIR,tally_ace_tests(7,:)/1e5)
plot(SNIR,tally_ace_tests(12,:)/1e5,'--')
plot(SNIR,tally_ace_tests(11,:)/1e5,'--')
plot(SNIR,tally_ace_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N, a = 0.5','K = 4*N, a = 0.5','K = 5*N, a = 0.5','K = 3*N, a = 4.5','K = 4*N, a = 4.5','K = 5*N, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - ACE - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_amf_tests(10,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(9,:)/1e5)
plot(SNIR,tally_amf_tests(7,:)/1e5)
plot(SNIR,tally_amf_tests(12,:)/1e5,'--')
plot(SNIR,tally_amf_tests(11,:)/1e5,'--')
plot(SNIR,tally_amf_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N, a = 0.5','K = 4*N, a = 0.5','K = 5*N, a = 0.5','K = 3*N, a = 4.5','K = 4*N, a = 4.5','K = 5*N, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - AMF - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_glrt_tests(10,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(9,:)/1e5)
plot(SNIR,tally_glrt_tests(7,:)/1e5)
plot(SNIR,tally_glrt_tests(12,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(11,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(6,:)/1e5,'--')
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N, a = 0.5','K = 4*N, a = 0.5','K = 5*N, a = 0.5','K = 3*N, a = 4.5','K = 4*N, a = 4.5','K = 5*N, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Sample Support - GLRT - K'), 'pdf') %Save figure

% %% For each detector, plot the probability of detection vs SNIR for different values of N (K held at 5*N) 
% %Plot the Probability of Detection vs the SNIR for ACE detector at N =
% %5,15,25, K = 5*N, P_fa = 1e-3
% figure
% plot(SNIR,tally_ace_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_ace_tests(5,:)/1e5)
% plot(SNIR,tally_ace_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5,15,25, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('N = 5','N = 15','N = 25','Location','southeast')   
% 
% %Plot the Probability of Detection vs the SNIR for AMF detector at N =
% %5,15,25, K = 5*N, P_fa = 1e-3
% figure
% plot(SNIR,tally_amf_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_amf_tests(5,:)/1e5)
% plot(SNIR,tally_amf_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5,15,25, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('N = 5','N = 15','N = 25','Location','southeast')   
% 
% %Plot the Probability of Detection vs the SNIR for GLRT detector at N =
% %5,15,25, K = 5*N, P_fa = 1e-3
% figure
% plot(SNIR,tally_glrt_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_glrt_tests(5,:)/1e5)
% plot(SNIR,tally_glrt_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5,15,25, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('N = 5','N = 15','N = 25','Location','southeast') 
% 
% 
%% For each detector, plot the probability of detection vs SNIR for different values of rho at a given N and K
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
%K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.1
figure
plot(SNIR,tally_ace_tests(2,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(15,:)/1e5)
plot(SNIR,tally_ace_tests(16,:)/1e5)
plot(SNIR,tally_ace_tests(4,:)/1e5,'--')
plot(SNIR,tally_ace_tests(13,:)/1e5,'--')
plot(SNIR,tally_ace_tests(14,:)/1e5,'--')
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9, a = 0.5','\rho = 0.5, a = 0.5','\rho = 0.1, a = 0.5','\rho = 0.9, a = 4.5','\rho = 0.5, a = 4.5','\rho = 0.1, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - ACE - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
%K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.13
figure
plot(SNIR,tally_amf_tests(2,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(15,:)/1e5)
plot(SNIR,tally_amf_tests(16,:)/1e5)
plot(SNIR,tally_amf_tests(4,:)/1e5,'--')
plot(SNIR,tally_amf_tests(13,:)/1e5,'--')
plot(SNIR,tally_amf_tests(14,:)/1e5,'--')
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9, a = 0.5','\rho = 0.5, a = 0.5','\rho = 0.1, a = 0.5','\rho = 0.9, a = 4.5','\rho = 0.5, a = 4.5','\rho = 0.1, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - AMF - K'), 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N =
%5,15,25, K = 5*N, P_fa = 1e-3
figure
plot(SNIR,tally_glrt_tests(2,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(15,:)/1e5)
plot(SNIR,tally_glrt_tests(16,:)/1e5)
plot(SNIR,tally_glrt_tests(4,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(13,:)/1e5,'--')
plot(SNIR,tally_glrt_tests(14,:)/1e5,'--')
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9, a = 0.5','\rho = 0.5, a = 0.5','\rho = 0.1, a = 0.5','\rho = 0.9, a = 4.5','\rho = 0.5, a = 4.5','\rho = 0.1, a = 4.5','Location','northwest','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Correlation - GLRT - K'), 'pdf') %Save figure

% %% For each detector, plot the probability of detection vs SNIR for different values of tau at a given N and K
% %Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
% %K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.1
% figure
% plot(SNIR,tally_ace_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_ace_tests(7,:)/1e5)
% plot(SNIR,tally_ace_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast')   
% 
% %Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
% %K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.13
% figure
% plot(SNIR,tally_amf_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_amf_tests(7,:)/1e5)
% plot(SNIR,tally_amf_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast')   
% 
% %Plot the Probability of Detection vs the SNIR for GLRT detector at N =
% %5,15,25, K = 5*N, P_fa = 1e-3
% figure
% plot(SNIR,tally_glrt_tests(4,:)/1e5)
% hold on
% plot(SNIR,tally_glrt_tests(7,:)/1e5)
% plot(SNIR,tally_glrt_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N'))
% xlabel('SNIR (dB)')
% ylabel('P_{d}')
% legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast') 
% 
% 
%% For each detector plot cos^2(theta) vs SNIR contour plots at constant Pd
file_name = 'Contour-Data-NG.mat';
full_file_name = fullfile(file_path2,file_name);
load(full_file_name)

figure
plot(contours_ng(:,1),cos2theta_contours_ng)
hold on
plot(contours_ng(:,4),cos2theta_contours_ng)
plot(contours_ng(:,7),cos2theta_contours_ng)
plot(contours_ng(:,10),cos2theta_contours_ng)
plot(contours_ng(:,13),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N, a = 0.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours - AMF - NG', 'pdf') %Save figure

figure
plot(contours_ng(:,2),cos2theta_contours_ng)
hold on
plot(contours_ng(:,5),cos2theta_contours_ng)
plot(contours_ng(:,8),cos2theta_contours_ng)
plot(contours_ng(:,11),cos2theta_contours_ng)
plot(contours_ng(:,14),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N, a = 0.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours - GLRT - NG', 'pdf') %Save figure

figure
plot(contours_ng(:,3),cos2theta_contours_ng)
hold on
plot(contours_ng(:,6),cos2theta_contours_ng)
plot(contours_ng(:,9),cos2theta_contours_ng)
plot(contours_ng(:,12),cos2theta_contours_ng)
plot(contours_ng(:,15),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N, a = 0.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours - ACE - NG', 'pdf') %Save figure


%% For each detector plot cos^2(theta) vs SNIR contour plots at constant Pd
file_name = 'Contour-Data-NG2.mat';
full_file_name = fullfile(file_path3,file_name);
load(full_file_name)

figure
plot(contours_ng(:,1),cos2theta_contours_ng)
hold on
plot(contours_ng(:,4),cos2theta_contours_ng)
plot(contours_ng(:,7),cos2theta_contours_ng)
plot(contours_ng(:,10),cos2theta_contours_ng)
plot(contours_ng(:,13),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N, a = 4.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours 2 - AMF - NG', 'pdf') %Save figure

figure
plot(contours_ng(:,2),cos2theta_contours_ng)
hold on
plot(contours_ng(:,5),cos2theta_contours_ng)
plot(contours_ng(:,8),cos2theta_contours_ng)
plot(contours_ng(:,11),cos2theta_contours_ng)
plot(contours_ng(:,14),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N, a = 4.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours 2 - GLRT - NG', 'pdf') %Save figure

figure
plot(contours_ng(:,3),cos2theta_contours_ng)
hold on
plot(contours_ng(:,6),cos2theta_contours_ng)
plot(contours_ng(:,9),cos2theta_contours_ng)
plot(contours_ng(:,12),cos2theta_contours_ng)
plot(contours_ng(:,15),cos2theta_contours_ng)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N, a = 4.5, b = 1'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\NG\Return Signal MA Contours 2 - ACE - NG', 'pdf') %Save figure
