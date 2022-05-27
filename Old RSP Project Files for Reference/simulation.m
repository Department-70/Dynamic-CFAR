%%%This file was used to generate and test the simulation environment that
%%%was later incorporated into the RL system. This file is not directly
%%%part of the RL system, but performs the same functions and generates
%%%plots for visualization.

clear all
%close all

%Define constants
fc = 77e9;
c = 3e8;
lambda = c/fc;

%%%Define Learnable Radar Parameters
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
bw = B(4);
Nsweep = N(3);

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
title('Signal'); axis tight;
subplot(212); spectrogram(sig,32,16,32,fs,'yaxis');
title('Signal spectrogram');

%Set distance to targets
target_dist1 = 0.5+(1-0.5)*rand();
target_dist2 = 1.1+(1.6-1.1)*rand();
target_dist3 = 5+(7-5)*rand();
%Set target velocities
target_speed1 = -1+(1-(-1))*rand();
target_speed2 = -1+(1-(-1))*rand();
target_speed3 = -2.5+(2.5-(-2.5))*rand();
%Define target behavior and motion model
targetprops = [target_dist1,target_speed1;target_dist2,target_speed2;...
    target_dist3,target_speed3];
target_rcs = 40;
target = phased.RadarTarget('MeanRCS',[target_rcs/8,target_rcs/4,...
    target_rcs*10],'PropagationSpeed',c,'OperatingFrequency',fc);
targetmotion = phased.Platform('InitialPosition',[[target_dist1;0;0],...
    [target_dist2;0;0],[target_dist3;0;0]],'Velocity',...
    [[target_speed1;0;0],[target_speed2;0;0],[target_speed3;0;0]]);

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
    txsig = repmat(txsig,1,3);

    % Propagate the signal and reflect off the target
    txsig = channel(txsig,radar_pos,tgt_pos,radar_vel,tgt_vel);
    txsig = target(txsig);

    % Dechirp the received radar return
    txsig = receiver(txsig);
    txsig = txsig(:,1)+txsig(:,2)+txsig(:,3);
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

%Plot the range-doppler response
clf;
plotResponse(rngdopresp,rs);                     
axis([-10 10 0 30])
clim = caxis;

%Define the guard and training band sizes for CFAR detector
guardsize = 2;
trainsize = 8;
totsize = 1+guardsize+trainsize;

%Pad the return signal array so that there are always 64 doppler bins.
padsize = (64-Nsweep)/2;
rs = padarray(rs,[0 padsize],0,'both');

%Define CFAR detector
cfar2D = phased.CFARDetector2D('GuardBandSize',guardsize,...
    'TrainingBandSize',trainsize,'ProbabilityFalseAlarm',0.0001e-4);

%Calculate the range-doppler response
[RSrngdop,rnggrid,dopgrid] = rngdopresp(rs);

%Determine the indices of the observed area
[~,rngindex] = min(abs(rnggrid-0));
rngindex = [rngindex, rngindex+256];
dopindex = [totsize, 64-totsize];
[colindex,rowindex] = meshgrid(dopindex(1):dopindex(2),...
  rngindex(1):rngindex(2));
index = [rowindex(:) colindex(:)]';

%Apply the CFAR detector to the range-doppler response and plot
detections = cfar2D(abs(RSrngdop).^2,index);
helperDetectionsMap(RSrngdop,rnggrid,dopgrid,rngindex,dopindex,detections)
axis([min(dopgrid) max(dopgrid) rnggrid(rngindex(1)) rnggrid(rngindex(2))])

%Apply image processing techniques to determine the location of the
%targets. First identify and number blobs, then determine the centroid
%locations.
detectionsmap = zeros(size(RSrngdop));
detectionsmap(rngindex(1):rngindex(2),dopindex(1):dopindex(2)) = ...
    reshape(double(detections),rngindex(2)-rngindex(1)+1,...
    dopindex(2)-dopindex(1)+1);
detectionblobs = bwlabel(detectionsmap,4);
s = regionprops(detectionblobs, 'centroid');
cent = cat(1,s.Centroid);
cent = sortrows(cent,2);
cent = round(cent);

%Determine the number of good/bad detections and use that to define the 
%reward function that was later used in the RL scenario
numdetected = max(max(detectionblobs));
reward = 0;
gooddetections = 0;
baddetections = abs(numdetected-length(targetprops));
for i=1:numdetected
    dettarrngdop = [rnggrid(cent(i,2)),-dopgrid(cent(i,1))];
    if(length(targetprops)==numdetected)
        error_vals = targetprops(i,:)-dettarrngdop;
    else
        [~,tempIndx] = min(abs(dettarrngdop-targetprops(:,1)));
        error_vals = targetprops(tempIndx(1),:)-dettarrngdop;
    end
    
    if(abs(error_vals(1))>0.1)
        baddetections = baddetections+1;
    elseif(abs(error_vals(2))>0.1)
        baddetections = baddetections+1;
    else
        gooddetections = gooddetections+1;
    end
end
if(gooddetections == 3)
    reward = reward+50;
end
reward = reward+gooddetections-baddetections+(16-Nsweep)/64;