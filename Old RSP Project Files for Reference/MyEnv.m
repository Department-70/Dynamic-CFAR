%%%This file defines my custom RL environment. It defines the environment's 
%%%properties and reset and step methods (including the reward and done 
%%%conditions). The reset function generates three new random targets. The
%%%step function performs the same functions as simulation.m, but without 
%%%generating any plots.

classdef MyEnv < rl.env.MATLABEnvironment    
    properties
        %Define constants
        fc = 77e9;
        c = 3e8;
        lambda = 3e8/77e9;

        %%%Define Learnable Radar Parameters
        %Bandwidth
        B = [1,2,3,4]*1e9;
        %Sweep Time (pulse duration)
        T = [10,20,30,40,50,60,70,80,90,100]*1e-6;
        %Chirp Repetition Time
        T_crt = [200,300,400,500,600,700,800,900]*1e-6;
        %Number of chirps
        N = [16,32,64];
        
        %Initialize target distances and velocities to 0
        target_dist1 = 0;
        target_dist2 = 0;
        target_dist3 = 0;
        target_speed1 = 0;
        target_speed2 = 0;
        target_speed3 = 0;
        
        %Define target RCS
        target_rcs = 40;
        
        %Define the radar system
        ant_aperture = 6.06e-4;                         
        ant_gain = 27;  
        tx_power = 3.2e-3;                     
        tx_gain = 36;                           
        rx_gain = 42;                         
        rx_nf = 4.5; 
        
        %Define the guard and training band sizes for CFAR detector
        guardsize = 2;
        trainsize = 8;
        totsize = 1+2+8;

        %Define CFAR detector
        cfar2D = phased.CFARDetector2D('GuardBandSize',2,...
            'TrainingBandSize',8,'ProbabilityFalseAlarm',0.0001e-4);
        
        %Define reward values
        RewardForAllDetections = 50;
        RewardPerGoodDetection = 1;
        PenaltyPerBadDetection = -3;
        PenaltyPerStep = -1; 
    end
    
    properties
        %Initialize state values.
        State = zeros(256,64)
    end
    
    properties(Access = protected)
        IsDone = false
    end
    
    
    methods
        %Define initializer function
        function this = MyEnv()
            %Initialize observation settings
            ObservationInfo = rlNumericSpec([256 64]);
            ObservationInfo.Name = 'Range-Doppler Map';
            ObservationInfo.Description = 'Matrix (range_bins x velocity_bins).';

            %Define action space - must define each unique combination and
            %save in a cell vector
            for i=1:4
                for j=1:10
                    for k=1:8
                        for l=1:3
                            act(l+(k-1)*3+(j-1)*24+(i-1)*240,:) = [i j k l];
                        end
                    end
                end
            end
            act = num2cell(act',[1 4]);
            
            %Initialize action settings   
            ActionInfo = rlFiniteSetSpec(act);
            ActionInfo.Name = 'Select Detection Settings';

            %Initialize built-in functions
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);            
        end
        
        %Define reset function - actions to be performed at the start of
        %each new episode
        function InitialObservation = reset(this)
            %Generate three new random targets
            this.target_dist1 = 0.5+(1-0.5)*rand();
            this.target_dist2 = 1.1+(1.6-1.1)*rand();
            this.target_dist3 = 5+(7-5)*rand();
            this.target_speed1 = -1+(1-(-1))*rand();
            this.target_speed2 = -1+(1-(-1))*rand();
            this.target_speed3 = -2.5+(2.5-(-2.5))*rand();
            %Reset range-doppler map to zeros
            InitialObservation = zeros(256,64);
            this.State = InitialObservation;            
        end
        
        %Define step function - advances the environment for each step
        %within an episode
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            IsDone = false;
            LoggedSignals = [];
            
            persistent n
            if isempty(n)
                n = 1;
            else
                n = n+1;
            end
            
            %Set learnable parameters based on learned Action
            B_cur = Action(1);
            T_cur = Action(2);
            T_crt_cur = Action(3);
            N_cur = Action(4);
            
            %Define parameters that depend on learned Actions
            tm = this.T(T_cur);
            CRI = this.T_crt(T_crt_cur);
            bw = this.B(B_cur);
            Nsweep = this.N(N_cur);
            sweep_slope = bw/tm;
            fs = bw;
            
            %Define waveform
            waveform = phased.LinearFMWaveform('SampleRate',fs,...
                'PulseWidth',tm,'PRF',1/(CRI),...
                'SweepBandwidth',bw,'SweepDirection','Up',...
                'Envelope','Rectangular',...
                'OutputFormat','Pulses','NumPulses',1);
            
            %Define target characteristics and motion model
            targetprops = [this.target_dist1,this.target_speed1;...
                this.target_dist2,this.target_speed2;this.target_dist3,...
                this.target_speed3];
            target = phased.RadarTarget('MeanRCS',[this.target_rcs/8,...
                this.target_rcs/4,this.target_rcs*10],...
                'PropagationSpeed',this.c,'OperatingFrequency',this.fc);
            targetmotion = phased.Platform('InitialPosition',...
                [[this.target_dist1;0;0],[this.target_dist2;0;0],...
                [this.target_dist3;0;0]],'Velocity',...
                [[this.target_speed1;0;0],[this.target_speed2;0;0],...
                [this.target_speed3;0;0]]);

            %Define propagation region (operation "scene")
            channel = phased.FreeSpace('PropagationSpeed',this.c,...
                'OperatingFrequency',this.fc,'SampleRate',fs,...
                'TwoWayPropagation',true);

            %Define radar system and motion
            transmitter = phased.Transmitter('PeakPower',this.tx_power,...
                'Gain',this.tx_gain);
            receiver = phased.ReceiverPreamp('Gain',this.rx_gain,...
                'NoiseFigure',this.rx_nf,'SampleRate',fs);
            radarmotion = phased.Platform('InitialPosition',[0;0;0],...
                'Velocity',[0;0;0]);
            
            %Initialize matrix for return signal
            rs = complex(zeros(round(waveform.SampleRate*CRI),Nsweep));

            %Transmit signal and measure target returns
            for m = 1:Nsweep
                % Update radar and target positions
                [radar_pos,radar_vel] = radarmotion(CRI);
                [tgt_pos,tgt_vel] = targetmotion(CRI);

                % Transmit FMCW waveform
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
            rngdopresp = phased.RangeDopplerResponse('RangeMethod',...
                'FFT','DopplerOutput','Speed','SweepSlope',sweep_slope,...
                'RangeFFTLengthSource','Property','RangeFFTLength',Nr,...
                'RangeWindow','Hann','DopplerFFTLengthSource',...
                'Property','DopplerFFTLength',Nd,'DopplerWindow','Hann',...
                'PropagationSpeed',this.c,'OperatingFrequency',this.fc,...
                'SampleRate',fs);
            
            %Pad the return signal array so that there are always 64 
            %doppler bins.
            padsize = (64-Nsweep)/2;
            rs = padarray(rs,[0 padsize],0,'both');
            
            %Calculate the range-doppler response
            [RSrngdop,rnggrid,dopgrid] = rngdopresp(rs);
            [~,rngindex] = min(abs(rnggrid-0));
            rngindex = [rngindex, rngindex+255];
            dopindex = [1, 64];
            
            %Set the Observation sent to the actor/critic networks equal to 
            %the range-doppler map
            Observation = abs(RSrngdop(rngindex(1):rngindex(2),...
                dopindex(1):dopindex(2)));

            %Determine the indices of the observed area
            [colindex,rowindex] = meshgrid(this.totsize:...
                dopindex(2)-this.totsize,rngindex(1):rngindex(2));
            index = [rowindex(:) colindex(:)]';

            %Apply the CFAR detector to the range-doppler response
            detections = this.cfar2D(abs(RSrngdop).^2,index);

            %Apply image processing techniques to determine the location of 
            %the targets. First identify and number blobs, then determine 
            %the centroid locations.
            detectionsmap = zeros(size(RSrngdop));
            detectionsmap(rngindex(1):rngindex(2),this.totsize:...
                dopindex(2)-this.totsize) = reshape(double(detections),...
                rngindex(2)-rngindex(1)+1,dopindex(2)-this.totsize-...
                this.totsize+1);
            detectionblobs = bwlabel(detectionsmap,4);
            s  = regionprops(detectionblobs, 'centroid');
            cent = cat(1,s.Centroid);
            cent = round(cent);
           
            %Determine the number of good/bad detections and use that to  
            %define the reward function
            numdetected = max(max(detectionblobs));
            gooddetections = 0;
            baddetections = abs(numdetected-length(targetprops));
            for i=1:numdetected
                dettarrngdop = [rnggrid(cent(i,2)),-dopgrid(cent(i,1))];
                [~,tempIndx] = min(abs(dettarrngdop-targetprops(:,1)));
                error_vals = targetprops(tempIndx(1),:)-dettarrngdop;
                
                if(abs(error_vals(1))>0.1)
                    baddetections = baddetections+1;
                elseif(abs(error_vals(2))>0.1)
                    baddetections = baddetections+1;
                else
                    gooddetections = gooddetections+1;
                end
            end
            
            if(n >= 10 || (gooddetections-baddetections) == 3)
                IsDone = true;
            end            
            reward = gooddetections*this.RewardPerGoodDetection+baddetections*this.PenaltyPerBadDetection+(16-Nsweep)/64+this.PenaltyPerStep;
            if(gooddetections == 3)
                Reward = reward+this.RewardForAllDetections;
            else
                Reward = reward;
            end            
        end
    
    end
    
end 