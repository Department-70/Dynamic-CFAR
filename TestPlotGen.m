close all
clear all

file_path = './Test Runs';
file_path2 = './Test Runs/Mismatched Targets';
file_path3 = './Test Runs/Homogeneity';
file_path4 = './Test Runs/Correlation PFA';
test_num = 55;
angle_test_num = 20;
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
    if i<10
        file_pattern = fullfile(file_path,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path,file_name);
        load(full_file_name)
    elseif i<30
        file_pattern = fullfile(file_path2,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path2,file_name);
        load(full_file_name)
        cos2theta_tests(i-9) = cos2theta;
    elseif i<46
        file_pattern = fullfile(file_path3,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path3,file_name);
        load(full_file_name)
    elseif i<56
        file_pattern = fullfile(file_path4,strcat('Test-',num2str(i),'-*.mat'));
        the_file = dir(file_pattern);
        file_name = the_file.name;
        full_file_name = fullfile(file_path4,file_name);
        load(full_file_name)
    end
    
    
    FA_ace_tests(i,:) = FA_ace;
    FA_amf_tests(i,:) = FA_amf;
    FA_glrt_tests(i,:) = FA_glrt;
    tally_ace_tests(i,:) = tally_ace;
    tally_amf_tests(i,:) = tally_amf;
    tally_glrt_tests(i,:) = tally_glrt;
    
    %Test over different values of K (training sample support)
    if i<4
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 25, K = 3*N,4*N,5*N, P_fa = 1e-3
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Sample Support: P_{d} vs SNIR')
%         subtitle(strcat('N = 25, K = ',num2str(2+i),'*N'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Support - N_25 - K_',num2str(2+i),'N - Gaussian'), 'pdf') %Save figure
        
    %Test over different values of N (signal length)
    elseif i<6
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5,15,25, K = 5*N P_fa = 1e-3       
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Sample Size: P_{d} vs SNIR')
%         subtitle(strcat('N = ',num2str(5*(2*(i-3)-1)),', K = 5*N'))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Size - N_',num2str(5*(2*(i-3)-1)),' - K_5N - Gaussian'), 'pdf') %Save figure
   
        if i==5
            figure
            plot(SNIR,tally_ace_tests(i-2,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-2,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-2,:)/1e5)
%             title('Impact of Sample Size: P_{d} vs SNIR')
%             subtitle(strcat('N = 25, K = 5*N'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold') 
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Size - N_25 - K_5N - Gaussian'), 'pdf') %Save figure
        end
        
    %Test over different values of tau for the sample under test (partially
    %homogeneous case)
    elseif i<8
        if i ==6
            %Plot the Probability of Detection vs the SNIR for each detector
            %at N = 5, K = 5*N P_fa = 1e-3 with tau = 1.0 (this is the
            %partially homogeneous Gaussian case)
            figure
            plot(SNIR,tally_ace_tests(i-2,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i-2,:)/1e5)
            plot(SNIR,tally_glrt_tests(i-2,:)/1e5)
%             title('Impact of Homogeneity: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, \tau = 1.0'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold')
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Homogeneity - N_5 - K_5N - tau_1_0 - Gaussian'), 'pdf') %Save figure
        end
            
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N P_fa = 1e-3 with tau = 1.3,1.5 (this is the
        %partially homogeneous Gaussian case)
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Homogeneity: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, \tau = ',num2str(1.7-0.1*(2*(i-5)))))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold') 
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Homogeneity - N_5 - K_5N - tau_1_',num2str(7-1*(2*(i-5))),' - Gaussian'), 'pdf') %Save
    
    %Test over different values of rho (one-lag correlation coefficient)
    elseif i<10
        if i==8
            %Plot the Probability of Detection vs the SNIR for each detector
            %at N = 5, K = 5*N P_fa = 1e-3 with rho = 0.9 (this shows the
            %impact of correlation)
            figure
            plot(SNIR,tally_ace_tests(i,:)/1e5)
            hold on
            plot(SNIR,tally_amf_tests(i,:)/1e5)
            plot(SNIR,tally_glrt_tests(i,:)/1e5)
%             title('Impact of Correlation: P_{d} vs SNIR')
%             subtitle(strcat('N = 5, K = 5*N, \rho = 0.9'))
            xlabel('SNIR (dB)','fontweight','bold')
            ylabel('P_{d}','fontweight','bold')
            legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold')
            set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
            set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
            saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation - N_5 - K_5N - rho_0_9 - Gaussian'), 'pdf') %Save
        end
        
        %Plot the Probability of Detection vs the SNIR for each detector
        %at N = 5, K = 5*N P_fa = 1e-3 with rho = 0.5,0.1 (this shows the
        %impact of correlation)
        figure
        plot(SNIR,tally_ace_tests(i,:)/1e5)
        hold on
        plot(SNIR,tally_amf_tests(i,:)/1e5)
        plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Correlation: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, \rho = ',num2str(0.5-0.4*(i-8))))
        xlabel('SNIR (dB)','fontweight','bold')
        ylabel('P_{d}','fontweight','bold')
        legend('ACE','AMF','GLRT','Location','southeast','fontweight','bold')
        set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
        saveas(gcf, strcat('C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation - N_5 - K_5N - rho_0_',num2str(5-4*(i-8)),' - Gaussian'), 'pdf') %Save
    
    %Test over different values of target return misalignment
%     elseif i<30
%         %Plot the Probability of Detection vs the SNIR for each detector
%         %at N = 5, K = 5*N P_fa = 1e-3 at a range of signal mismatch angles
%         figure
%         plot(SNIR,tally_ace_tests(i,:)/1e5)
%         hold on
%         plot(SNIR,tally_amf_tests(i,:)/1e5)
%         plot(SNIR,tally_glrt_tests(i,:)/1e5)
%         title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
%         subtitle(strcat('N = 5, K = 5*N, cos^{2}(\theta) = ',num2str(real(cos2theta_tests(i-9)))))
%         xlabel('SNIR (dB)')
%         ylabel('P_{d}')
%         legend('ACE','AMF','GLRT','Location','southeast')
    elseif (i>29) && (i<46)
        tau_tests(i-29,:) = tau;
    elseif (i>45) && (i<56)
        rho_tests(i-45,:) = rho;
    end
end


%% For each detector, plot the probability of false alarm vs rho.
figure
hold on
plot(rho_tests,log10(FA_ace_tests(46:55)/1e5))
plot(rho_tests,log10(FA_amf_tests(46:55)/1e5))
plot(rho_tests,log10(FA_glrt_tests(46:55)/1e5))
axis([0 1 -4 0])
yl = yline(-3,'--','P_{fa} = 10^{-3}');
yl.LabelHorizontalAlignment = 'left';
xlabel('One-Lag Clutter Correlation Coefficient (\rho)','fontweight','bold')
ylabel('log_{10}(P_{fa})','fontweight','bold')
box on
legend('ACE','AMF','GLRT','location','southeast','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation PFA - Wide - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of false alarm vs tau for the partially homogeneous case.
temp_ace = [flipud(FA_ace_tests(41:45,1)); FA_ace_tests(30:40,1)];
temp_glrt = [flipud(FA_glrt_tests(41:45,1)); FA_glrt_tests(30:40,1)];
temp_amf = [flipud(FA_amf_tests(41:45,1)); FA_amf_tests(30:40,1)];
temp_tau = [flipud(tau_tests(12:16)); tau_tests(1:11)];
figure
hold on
plot(temp_tau,log10(temp_ace/1e5))
plot(temp_tau,log10(temp_amf/1e5))
plot(temp_tau,log10(temp_glrt/1e5))
yl = yline(-3,'--','P_{fa} = 10^{-3}');
yl.LabelHorizontalAlignment = 'left';
xl = xline(1,'--','\tau = 1');
xl.LabelOrientation = 'horizontal';
xl.LabelVerticalAlignment = 'middle';
xlabel('Sample Under Test Clutter Intensity (\tau)','fontweight','bold')
ylabel('log_{10}(P_{fa})','fontweight','bold')
box on
legend('ACE','AMF','GLRT','location','southeast','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [7.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\PFA - Wide - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR with respect to signal return misalignment.
figure
hold on
for i=1:20
    plot(SNIR,tally_glrt_tests(9+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('GLRT Detector, N = 5, K = 5*N')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
x = [0.34 0.7];
y = [0.63 0.115];
box on
annotation('arrow',x,y)
annotation('textbox',[0.24 0.63 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA - GLRT - Gaussian', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_amf_tests(9+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('AMF Detector, N = 5, K = 5*N')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
x = [0.33 0.7];
y = [0.64 0.115];
box on
annotation('arrow',x,y)
annotation('textbox',[0.23 0.64 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA - AMF - Gaussian', 'pdf') %Save figure

figure
hold on
for i=1:20
    plot(SNIR,tally_ace_tests(9+i,:)/1e5)
end
% title('Impact of Return Signal Misalignment: P_{d} vs SNIR')
% subtitle('ACE Detector, N = 5, K = 5*N')
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
box on
x = [0.52 0.7];
y = [0.5 0.115];
annotation('arrow',x,y)
annotation('textbox',[0.42 0.5 0.18 0.07],'String','cos^{2}(\theta): 1 => 0')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA - ACE - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of K at a given N 
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_ace_tests(1,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(2,:)/1e5)
plot(SNIR,tally_ace_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N','K = 4*N','K = 5*N','Location','southeast','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Support - ACE - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_amf_tests(1,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(2,:)/1e5)
plot(SNIR,tally_amf_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N','K = 4*N','K = 5*N','Location','southeast','fontweight','bold')  
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Support - AMF - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 25,
%K = 3*N,4*N,5*N, P_fa = 1e-3
figure
plot(SNIR,tally_glrt_tests(1,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(2,:)/1e5)
plot(SNIR,tally_glrt_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 25, K = 3,4,5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('K = 3*N','K = 4*N','K = 5*N','Location','southeast','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Support - GLRT - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of N (K held at 5*N) 
%Plot the Probability of Detection vs the SNIR for ACE detector at N =
%5,15,25, K = 5*N, P_fa = 1e-3
figure
plot(SNIR,tally_ace_tests(4,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(5,:)/1e5)
plot(SNIR,tally_ace_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5','N = 15','N = 25','Location','southeast','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Size - ACE - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N =
%5,15,25, K = 5*N, P_fa = 1e-3
figure
plot(SNIR,tally_amf_tests(4,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(5,:)/1e5)
plot(SNIR,tally_amf_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5','N = 15','N = 25','Location','southeast','fontweight','bold')   
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Size - AMF - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N =
%5,15,25, K = 5*N, P_fa = 1e-3
figure
plot(SNIR,tally_glrt_tests(4,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(5,:)/1e5)
plot(SNIR,tally_glrt_tests(3,:)/1e5)
% title('Impact of Sample Support: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5,15,25, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('N = 5','N = 15','N = 25','Location','southeast','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Sample Size - GLRT - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of rho at a given N and K
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
%K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.1
figure
plot(SNIR,tally_ace_tests(4,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(8,:)/1e5)
plot(SNIR,tally_ace_tests(9,:)/1e5)
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9','\rho = 0.5','\rho = 0.1','Location','southeast','fontweight','bold')
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation - ACE - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
%K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.13
figure
plot(SNIR,tally_amf_tests(4,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(8,:)/1e5)
plot(SNIR,tally_amf_tests(9,:)/1e5)
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9','\rho = 0.5','\rho = 0.1','Location','southeast','fontweight','bold')   
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation - AMF - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 5,
%K = 5*N, P_fa = 1e-3, rho = 0.9,0.5,0.1
figure
plot(SNIR,tally_glrt_tests(4,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(8,:)/1e5)
plot(SNIR,tally_glrt_tests(9,:)/1e5)
% title('Impact of Covariance: P_{d} vs SNIR')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\rho = 0.9','\rho = 0.5','\rho = 0.1','Location','southeast','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Correlation - GLRT - Gaussian', 'pdf') %Save figure


%% For each detector, plot the probability of detection vs SNIR for different values of tau at a given N and K
%Plot the Probability of Detection vs the SNIR for ACE detector at N = 5,
%K = 5*N, P_fa = 1e-3, tau = 1.0,1.3,1.5
figure
plot(SNIR,tally_ace_tests(4,:)/1e5)
hold on
plot(SNIR,tally_ace_tests(7,:)/1e5)
plot(SNIR,tally_ace_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast','fontweight','bold')   
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Homogeneity - ACE - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for AMF detector at N = 5,
%K = 5*N, P_fa = 1e-3, tau = 1.0,1.3,1.5
figure
plot(SNIR,tally_amf_tests(4,:)/1e5)
hold on
plot(SNIR,tally_amf_tests(7,:)/1e5)
plot(SNIR,tally_amf_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast','fontweight','bold')   
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Homogeneity - AMF - Gaussian', 'pdf') %Save figure

%Plot the Probability of Detection vs the SNIR for GLRT detector at N = 5,
%K = 5*N, P_fa = 1e-3, tau = 1.0,1.3,1.5
figure
plot(SNIR,tally_glrt_tests(4,:)/1e5)
hold on
plot(SNIR,tally_glrt_tests(7,:)/1e5)
plot(SNIR,tally_glrt_tests(6,:)/1e5)
% title('Performance in Partially Homogeneous Gaussian Clutter')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('P_{d}','fontweight','bold')
legend('\tau = 1.0','\tau = 1.3','\tau = 1.5','Location','southeast','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Homogeneity - GLRT - Gaussian', 'pdf') %Save figure


%% For each detector plot cos^2(theta) vs SNIR contour plots at constant Pd
file_name = 'Contour-Data-Gaussian.mat';
full_file_name = fullfile(file_path2,file_name);
load(full_file_name)

figure
plot(contours(:,1),cos2theta_contours)
hold on
plot(contours(:,4),cos2theta_contours)
plot(contours(:,7),cos2theta_contours)
plot(contours(:,10),cos2theta_contours)
plot(contours(:,13),cos2theta_contours)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('AMF Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA Contours - AMF - Gaussian', 'pdf') %Save figure

figure
plot(contours(:,2),cos2theta_contours)
hold on
plot(contours(:,5),cos2theta_contours)
plot(contours(:,8),cos2theta_contours)
plot(contours(:,11),cos2theta_contours)
plot(contours(:,14),cos2theta_contours)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('GLRT Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA Contours - GLRT - Gaussian', 'pdf') %Save figure

figure
plot(contours(:,3),cos2theta_contours)
hold on
plot(contours(:,6),cos2theta_contours)
plot(contours(:,9),cos2theta_contours)
plot(contours(:,12),cos2theta_contours)
plot(contours(:,15),cos2theta_contours)
axis([0 35 0 1])
% title('Rejection of Misaligned Target Returns')
% subtitle(strcat('ACE Detector, N = 5, K = 5*N'))
xlabel('SNIR (dB)','fontweight','bold')
ylabel('cos^{2}(\theta)','fontweight','bold')
legend('P_{d} = 0.1','P_{d} = 0.25','P_{d} = 0.5','P_{d} = 0.75','P_{d} = 0.9','Location','southwest','fontweight','bold') 
set(gcf, 'PaperPosition', [-0.2 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5.4 5]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'C:\Users\Alex\Desktop\General Exam\Implementation\Plots\Gaussian\Return Signal MA Contours - ACE - Gaussian', 'pdf') %Save figure

