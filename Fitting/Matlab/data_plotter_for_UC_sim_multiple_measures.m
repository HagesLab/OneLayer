% Import raw and simulation data for UC project and plot together, allowing
% quick assessment of goodness of fit (optional - calculate and display
% numerical evaluation of goodness of fit).

close all; clear all;


% Data repositories are selected at top of script, the result should follow
% automatically. Just make sure each directory contains only the relevant
% files. Alternatively, use wildcard search in the dir function calls to select
% only the relevant files.

% Variables are appended with suffix indicated exp or sim origin

% Data import setup. Two methods are possible for data import: 'dir' method
% searches current folder (or other folders) for matching filenames, and
% builds a structure 'list.name' containing file names. 'direct' method
% uses a list of filenames provided directly in the script. Select the
% method used with the switch below:

% Multiple arrays of sim data can be selected by concatanating address
% strings in dir_sim

% Use 'data_plotter_for_UC_sim_with_arrays' for comparing multiple sets of simulation data
% for a particular PL measurement with the relnvant expeirmental data. Use
% this script for comparing % e.g. UC and pvsk PL data from a single sample with the equivalent
% simulated dataset. Suffix indicates relevant measurement, e.g.
% dir_exp_TRPL_UC

% SAMPLE DESIGNATION
sample = 'S2';

% INIT FILE DIRECTORY
dir_file_init = 'D:\Dropbox\$Main\k. Sims\OneLayer-master\Initial\20210817 S 50\';

% INIT FILE NAME
file_init = '__Power_7.0000e-01.txt'

% SIM PVSK DIRECTORY
dir_sim_TRPL_pvsk = 'D:\Dropbox\$Main\k. Sims\OneLayer-master\Analysis\20210818 S2 batch'; 

% SIM UC DIRECTORY
dir_sim_TRPL_UC = 'D:\Dropbox\$Main\k. Sims\OneLayer-master\Analysis\20210818 S2 batch'; 

% EXP PVSK DIRECTORY
dir_exp_TRPL_pvsk = 'D:\Dropbox\$Main\l. Supervision\Prashan\Data\FAMAPI UC data for simulation Aug-2021\20210728 FAMAPI UC\S2\FAMAPI TRPL';

% EXP UC DIRECTORY
dir_exp_TRPL_UC = 'D:\Dropbox\$Main\l. Supervision\Prashan\Data\FAMAPI UC data for simulation Aug-2021\20210728 FAMAPI UC\S2\UC TRPL';

t_offset_sim = 254;  % t offset of sim data to exp t0

normtime = 350;  % normalization time point in ns wrt t_exp (must come after t_offset_sim)

UC_norm_onset = 433;  % time (in absolute units) after which UC transient should be normalized to max. (to avoid laser artefact at t0)
    
%%%%%%%%%%%%%%%%%%%%%
%% Import exp data %%
%%%%%%%%%%%%%%%%%%%%%
% The listc function sorts sensibly, this is retained for future
% developments but is not currently utilized

list_exp_TRPL_pvsk=dir(strcat(dir_exp_TRPL_pvsk,'\*.dat'));  % create structure that holds filenames
listc_exp_TRPL_pvsk = {list_exp_TRPL_pvsk(:).name};
listc_exp_TRPL_pvsk = sort(listc_exp_TRPL_pvsk);  % sort entries in listc sensibly

list_exp_TRPL_UC = dir(strcat(dir_exp_TRPL_UC,'\*.dat'));
listc_exp_TRPL_UC = {list_exp_TRPL_UC(:).name};
listc_exp_TRPL_UC = sort(listc_exp_TRPL_UC);


% Import data and generate individual time vectors, since rep. rate and
% time res. can change depending on the sample

stack_exp_TRPL_pvsk = NaN(32768,numel(list_exp_TRPL_pvsk));  % empty matrix to hold imported transients
t_exp_TRPL_pvsk = stack_exp_TRPL_pvsk;  % empty time vector array

stack_exp_TRPL_UC = NaN(32768,numel(list_exp_TRPL_UC));  % empty matrix to hold imported transients
t_exp_TRPL_UC = stack_exp_TRPL_UC;  % empty time vector array

%%%%%%%%%%%%%%%%%%
%%% Import and perform adjustments on individual exp transients
%% PVSK exp data

for i=1:numel(list_exp_TRPL_pvsk)  % iterate over number of transients
    

    stack_exp_TRPL_pvsk(:,i) = dlmread(fullfile(list_exp_TRPL_pvsk(i).folder, list_exp_TRPL_pvsk(i).name),'.',10,0);  % import raw data into 'stack'

    tres = dlmread(fullfile(list_exp_TRPL_pvsk(i).folder, list_exp_TRPL_pvsk(i).name),'.',[8 0 8 0]);  % Time res in ns
    
    t_exp_TRPL_pvsk(:,i) = linspace(0,32768*tres,32768);
    
    % Trim stack and t vector
    
    % Baseline adjustment
    cut=150;  % time in ns at which baseline end is defined
    [~,cut] = min(abs(cut-t_exp_TRPL_pvsk));
    stack_exp_TRPL_pvsk(:,i) = stack_exp_TRPL_pvsk(:,i)-mean(stack_exp_TRPL_pvsk(1:cut,i));  % subtracts pre-t0 baseline
        
    % normalise by dithered maximum amplitude point
    % identify time index after UC_norm_onset (typically not necessary for
    % PVSK transients)
%     
%     [~,index] = min(UC_norm_onset - t_exp_TRPL_pvsk(:,i));  % find t point index for normalization
%     
%     [~,index2] = max(stack_exp_TRPL_pvsk(index:end,i));  % index2 corresponds to maximum of UC transient after UC_norm_onset
%         
%     stack_exp_TRPL_pvsk(:,i) = stack_exp_TRPL_pvsk(:,i)/mean(stack_exp_TRPL_pvsk(index2-1:index2+1,i));  % normalize by dithered max. at index2
%     
    % Normalise by time point span away from t0 (to avoid possible t0 time res
    % artefacts)
    
    [~,index] = min(abs(t_exp_TRPL_pvsk - normtime));
    stack_exp_TRPL_pvsk(:,i) = stack_exp_TRPL_pvsk(:,i)/mean(stack_exp_TRPL_pvsk(index-1:index+1,i));
    
    
    % smoothing (uncomment if required)
    %sstart=2400; % index point to start smoothing
    %[~,index]=min(abs(sstart-t));
    %stack(600:end,i)=smooth(stack(600:end,i),15);
    
end

%%%%%%%%%%%%%%%%%%
%% UC exp data

for i=1:numel(list_exp_TRPL_UC)  % iterate over number of UC transients
    
    stack_exp_TRPL_UC(:,i) = dlmread(fullfile(list_exp_TRPL_UC(i).folder, list_exp_TRPL_UC(i).name),'.',10,0);  % import raw data into 'stack'

    tres = dlmread(fullfile(list_exp_TRPL_UC(i).folder, list_exp_TRPL_UC(i).name),'.',[8 0 8 0]);  % Time res in ns
    
    t_exp_TRPL_UC(:,i) = linspace(0,32768*tres,32768);
    
    % Trim stack and t vector
    
    % Baseline adjustment
    cut=150;  % time in ns at which baseline end is defined
    [~,cut] = min(abs(cut-t_exp_TRPL_UC(:,i)));
    stack_exp_TRPL_UC(:,i) = stack_exp_TRPL_UC(:,i)-mean(stack_exp_TRPL_UC(1:cut,i));  % subtracts pre-t0 baseline
        
    % normalise by dithered maximum amplitude point
    % identify time index after UC_norm_onset
    
    [~,index] = min(abs(UC_norm_onset - t_exp_TRPL_UC(:,i)));  % find t point index for normalization
    
    [~,index2] = max(stack_exp_TRPL_UC(index:end,i));  % index2 corresponds to maximum of UC transient after UC_norm_onset
    index2 = index2 + index;  % adjust index2 to correct position on non-truncated vector
   
    stack_exp_TRPL_UC(:,i) = stack_exp_TRPL_UC(:,i)/mean(stack_exp_TRPL_UC(index2-5:index2+5,i));  % normalize by dithered max. at index2
    %stack_exp_TRPL_UC(:,i) = stack_exp_TRPL_UC(:,i)/max(stack_exp_TRPL_UC(index:end,i)); 
    
    % Normalise by time point span away from t0 (to avoid possible t0 time res
    % artefacts)
    
    %[~,index] = min(abs(t_exp_TRPL_UC - normtime));
    %stack_exp_TRPL_UC(:,i) = stack_exp_TRPL_UC(:,i)/mean(stack_exp_TRPL_UC(index-1:index+1,i));
    
    % smoothing (uncomment if required)
    sstart=index; % index point to start smoothing
    stack_exp_TRPL_UC(index:end,i)=smooth(stack_exp_TRPL_UC(index:end,i),15);
    
end

% plot(t_exp_TRPL_UC,stack_exp_TRPL_UC)

%%% Import simulation data for pvsk and UC following same logic
% Note that data can be exported only as one file per measurement. Series
% are independent and can have different formats.

%% 1. PVSK TRPL sim data
    
    % Use wildcard search to select appropriate file if necessary
    list_sim_TRPL_pvsk = dir(strcat(dir_sim_TRPL_pvsk,'\*PVSK*.csv'));  % create structure that holds filenames
    import_sim_TRPL_pvsk = dlmread(fullfile(list_sim_TRPL_pvsk.folder, list_sim_TRPL_pvsk.name),',',1,0);

    % Trim imported data file to yield stack
    stack_sim_TRPL_pvsk = import_sim_TRPL_pvsk(:,2:end);
    
    t_sim_TRPL_pvsk = import_sim_TRPL_pvsk(:,1) + t_offset_sim;  % assumes same t over all sim transients

    % Normalize transient amplitudes
    for j=1:numel(stack_sim_TRPL_pvsk(1,:))
                
        % Normalize each transient to value at normtime (after applying
        % t_offset_sim)
        
        [~,index] = min(abs(t_sim_TRPL_pvsk - normtime));
        stack_sim_TRPL_pvsk(:,j) = stack_sim_TRPL_pvsk(:,j)/mean(stack_sim_TRPL_pvsk(index-1:index+1,j));
        
        % Normalization by maximum height, at implied t0
        % stack_sim_TRPL_pvsk(:,j) = stack_sim_TRPL_pvsk(:,j)/max(stack_sim_TRPL_pvsk(:,j));

    end
    
%% 2. UC TRPL sim data
    
    % Use wildcard search to select appropriate file if necessary
    list_sim_TRPL_UC = dir(strcat(dir_sim_TRPL_UC,'\*UC*.csv'));  % create structure that holds filenames
    import_sim_TRPL_UC = dlmread(fullfile(list_sim_TRPL_UC.folder, list_sim_TRPL_UC.name),',',1,0);

    % Trim imported data file to yield stack
    stack_sim_TRPL_UC = import_sim_TRPL_UC(:,2:end);
    
    t_sim_TRPL_UC = import_sim_TRPL_UC(:,1) + t_offset_sim;  % assumes same t over all sim transients

    % Normalize transient amplitudes
    for j=1:numel(stack_sim_TRPL_UC(1,:))
                
        % Normalize each transient to value at normtime (after applying
        % t_offset_sim)
        
        %[~,index] = min(abs(t_sim_TRPL_UC - normtime));
        %stack_sim_TRPL_UC(:,j) = stack_sim_TRPL_UC(:,j)/mean(stack_sim_TRPL_UC(index-1:index+1,j));
        
        % Normalization by maximum height, at implied t0
        stack_sim_TRPL_UC(:,j) = stack_sim_TRPL_UC(:,j)/max(stack_sim_TRPL_UC(:,j));

    end
    
    %semilogy(t_sim_TRPL_pvsk,stack_sim_TRPL_pvsk);
    %semilogy(t_sim_TRPL_UC,stack_sim_TRPL_UC);


%%%%%%%%%%%%%%%%%%%%%%%%
%% Display results
%%%%%%%%%%%%%%%%%%%%%%%%

% % Single plot displaying all results
%  semilogy(t_exp_TRPL_pvsk,stack_exp_TRPL_pvsk,'LineWidth',0.25); hold on;
%  for i = 1:numel(dir_sim_TRPL_pvsk(:,1))
%     semilogy(t_sim,stack_sim(:,:,i),'k','LineWidth',1);
%  end
f = figure;
subplot(2,2,1)
loglog(t_exp_TRPL_pvsk,stack_exp_TRPL_pvsk,'LineWidth',0.25); hold on;
loglog(t_sim_TRPL_pvsk,stack_sim_TRPL_pvsk,'k--','LineWidth',0.5);
xlim([t_offset_sim 6000]);
ylim([1E-3 5]);
xlabel('time (ns)'); ylabel('intensity');
title(strcat(sample,'--','PVSK TRPL'));

subplot(2,2,2)
semilogx(t_exp_TRPL_UC,stack_exp_TRPL_UC,'LineWidth',0.25); hold on;
semilogx(t_sim_TRPL_UC,stack_sim_TRPL_UC(:,1:3),'k--','LineWidth',0.5);
xlim([t_offset_sim 20000])
ylim([1E-2 1]);
xlabel('time (ns)'); ylabel('intensity');
title(strcat(sample,'--','UC TRPL'));

%% Extract and append text from initial file to figure

init_text = readtable(strcat(dir_file_init,file_init));

T = init_text;  % debug

% Split table into two halves
T1 = init_text(1:floor(height(init_text)/3),:);
T2 = init_text(ceil(height(init_text)/3):floor(height(init_text)*(2/3)),:);
T3 = init_text(ceil(height(init_text)/3)*2:end,:);

% Print two halves of table on bottom-right corner of figure
uitable('Data',T1{:,:},'ColumnName',T1.Properties.VariableNames,'ColumnWidth',{110},...
   'RowName',T1.Properties.RowNames,'Units', 'Normalized','Position',[0.5, 0, 0.15, 0.5]);

uitable('Data',T2{:,:},'ColumnName',T2.Properties.VariableNames,'ColumnWidth',{110},...
    'RowName',T2.Properties.RowNames,'Units','Normalized','Position',[0.65, 0, 0.15, 0.5]);

uitable('Data',T3{:,:},'ColumnName',T3.Properties.VariableNames,'ColumnWidth',{110},...
    'RowName',T3.Properties.RowNames,'Units','Normalized','Position',[0.8, 0, 0.15, 0.5]);
 
%%
set(gcf, 'Position', get(0, 'Screensize'));
% Save figure for export and reference

temp = datestr(now, 30);
print([sample, ' ', temp,' ','RWM'], '-dpng', '-r300')
close all

%% Calculate and plot differential lifetimes from exp data
% tau_diff = NaN(size(stack_exp_TRPL_pvsk));
% step = 1;
% 
% % Find index of tau0
% [~,index] = max(stack_exp_TRPL_pvsk(:,1));
% 
% for i = 1:numel(list_exp)
%     for j=index:step:numel(stack_exp_TRPL_pvsk(:,1))-step-1
%         tau_diff(j,i) = -((log(stack_exp_TRPL_pvsk(j+step,i))-log(stack_exp_TRPL_pvsk(j,i)))/(t_exp_TRPL_pvsk(j+step,i)-t_exp_TRPL_pvsk(j,i)))^-1;
% %         clc
% %         disp(stack_exp(j,i));
% %         disp(stack_exp(j+1,i));
% %         disp((t_exp(j+1,i)-t_exp(j,i)));
% %         disp(tau_diff(j,i));
% %         pause()
%         
%     end
% end
% 
% figure
% semilogy(t_exp_TRPL_pvsk(:,:),tau_diff(:,:))
% 
% %% Calculate and plot differential lifetimes from sim data
% tau_diff_sim = NaN(size(stack_sim));
% step = 1;
% for i = 1:numel(stack_sim(1,:))
%     for j=1:step:numel(stack_sim(:,1))-1
%         
%         tau_diff_sim(j,i) = -((log(stack_sim(j+step,i))-log(stack_sim(j,i)))/(t_sim(j+step)-t_sim(j)))^-1;
% %         clc
% %         disp(stack_exp(j,i));
% %         disp(stack_exp(j+1,i));
% %         disp((t_exp(j+1,i)-t_exp(j,i)));
% %         disp(tau_diff(j,i));
% %         pause()
%         
%     end
% end
% 
% figure
% semilogy(t_sim,tau_diff_sim)


