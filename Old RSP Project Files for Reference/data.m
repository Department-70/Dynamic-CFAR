%%%This file is used to generate training data to pretraining the CNN. Some
%%%modification is required for each desired scenario (chaning the number
%%%of targets being generated/detected).

clear all

%Can set these to allow for reproducibility
%rng(2021); %Three target scenario - test1.mat
%rng(2012);  %Two targets close - test4.mat
%rng(3000);  %Two targets far - test5.mat
%rng(3021);  %One target near - test6.mat
rng(4021);  %One target far - test7.mat

%Define constants
fc = 77e9;
c = 3e8;
lambda = c/fc;

%%%Define learnable Radar Parameters
%Bandwidth
B = [1,2,3,4]*1e9;
%Sweep Time (pulse duration)
T = [10,20,30,40,50,60,70,80,90,100]*1e-6;
%Chirp Repetition Time
T_crt = [200,300,400,500,600,700,800,900]*1e-6;
%Number of chirps
N = [16,32,64];

%Select bandwidth, sweep time, chirp repetition time, and number of chirps
tm = T(10);
CRI = T_crt(1);
bw = B(1);
Nsweep = N(2);

%Define sweep slope
sweep_slope = bw/tm;
%Define sampling frequency
fs = bw;

%Generate waveform
waveform = phased.LinearFMWaveform('SampleRate',fs,...
    'PulseWidth',tm,'PRF',1/(CRI),...
    'SweepBandwidth',bw,'SweepDirection','Up',...
    'Envelope','Rectangular',...
    'OutputFormat','Pulses','NumPulses',1);

%Define and plot signal
sig = waveform();
subplot(211); plot(0:1/fs:CRI-1/fs,real(sig));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('FMCW signal'); axis tight;
subplot(212); spectrogram(sig,32,16,32,fs,'yaxis');
title('FMCW signal spectrogram');

%Run through loop to generate training data samples
for i=1:1000
    %Print current loop number to track progress
    i
    %Set distance to targets
    target_dist1 = 5+(7-5)*rand();
    %Set target velocities
    target_speed1 = -3.5+(3.5-(-3.5))*rand();
    %Define target behavior and motion model
    targetprops = [target_dist1,target_speed1];
    target_rcs = 40;
    target = phased.RadarTarget('MeanRCS',target_rcs/8,...
        'PropagationSpeed',c,'OperatingFrequency',fc);
    targetmotion = phased.Platform('InitialPosition',...
        [[target_dist1;0;0]],'Velocity',[[target_speed1;0;0]]);

    %Define propagation region (operation "scene")
    channel = phased.FreeSpace('PropagationSpeed',c,...
        'OperatingFrequency',fc,'SampleRate',fs,'TwoWayPropagation',true);

    
    %Define the radar system
    ant_aperture = 6.06e-4;                         
    ant_gain = 27;  
    tx_power = 3.2e-3;                     
    tx_gain = 36;                           
    rx_gain = 42;                         
    rx_nf = 4.5;
    transmitter = phased.Transmitter('PeakPower',tx_power,'Gain',tx_gain);
    receiver = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
        'SampleRate',fs);

    %Define radar motion (radar is stationary)
    radarmotion = phased.Platform('InitialPosition',[0;0;0],...
        'Velocity',[0;0;0]);

    %Initialize matrix for return signal
    rs = complex(zeros(round(waveform.SampleRate*CRI),Nsweep));

    %Transmit signal and measure target returns
    for m = 1:Nsweep
        % Update radar and target positions
        [radar_pos,radar_vel] = radarmotion(CRI);
        [tgt_pos,tgt_vel] = targetmotion(CRI);

        % Transmit waveform
        sig = waveform();
        txsig = transmitter(sig);

        % Propagate the signal and reflect off the target
        txsig = channel(txsig,radar_pos,tgt_pos,radar_vel,tgt_vel);
        txsig = target(txsig);

        % Dechirp the received radar return
        txsig = receiver(txsig);
        dechirpsig = dechirp(txsig,sig);

        rs(:,m) = dechirpsig;
    end

    %Define range-doppler response
    Nft = round(CRI*waveform.SampleRate);                               
    Nst = 64;
    Nr = 2^nextpow2(Nft);                         
    Nd = 2^nextpow2(Nst); 
    rngdopresp = phased.RangeDopplerResponse('RangeMethod','FFT',...
        'DopplerOutput','Speed','SweepSlope',sweep_slope,...
        'RangeFFTLengthSource','Property','RangeFFTLength',Nr,...
        'RangeWindow','Hann',...
        'DopplerFFTLengthSource','Property','DopplerFFTLength',Nd,...
        'DopplerWindow','Hann',...
        'PropagationSpeed',c,'OperatingFrequency',fc,'SampleRate',fs);

    %Pad the return signal array so that there are always 64 doppler bins.
    padsize = (64-Nsweep)/2;
    rs = padarray(rs,[0 padsize],0,'both');

    %Calculate the range-doppler response
    [RSrngdop,rnggrid,dopgrid] = rngdopresp(rs);

    %Determine the indices of the observed area
    [~,rngindex] = min(abs(rnggrid-0));
    rngindex = [rngindex, rngindex+255];
    dopindex = [1, 64];

    %Store output values in matrices ready to be saved to a file
    outmap(i,:,:) = RSrngdop(rngindex(1):rngindex(2),...
        dopindex(1):dopindex(2));
    outrnggrid(i,:) = rnggrid(rngindex(1):rngindex(2));
    outdopgrid(i,:) = dopgrid(dopindex(1):dopindex(2));
    outtruth(i,:,:) = targetprops;
end


%Determine the indices of the true target positions as training labels
for i=1:1000
    for j=1:1
        [~,truerngindex] = min(abs(outrnggrid(i,:)-outtruth(i,j,1)));
        [~,truedopindex] = min(abs(outdopgrid(i,:)-outtruth(i,j,2)));
        outindex(i,j,:) = [truerngindex, truedopindex];
    end
end

%Save training data and labels to a file
filename = 'test8.mat';
save(filename,'outmap','outrnggrid','outdopgrid','outtruth','outindex')