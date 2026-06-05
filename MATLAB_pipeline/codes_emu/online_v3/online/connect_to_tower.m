function [success, tower_ip, ssh_conn] = connect_to_tower(varargin)
% CONNECT_TO_TOWER - Attempt to connect to Tower computer for offline processing
%
% Tries multiple IP addresses to establish SSH connection to Tower.
%
% Usage:
%   [success, tower_ip, ssh_conn] = connect_to_tower()
%   [success, tower_ip, ssh_conn] = connect_to_tower('timeout', 5)
%
% Outputs:
%   success   - Boolean indicating if connection was successful
%   tower_ip  - IP address that worked (empty if failed)
%   ssh_conn  - SSH connection string for subsequent commands
%
% Tower IP addresses to try (in order):
%   R26-1: 192.168.42.226
%   R26-2: 192.168.137.226
%   R28-1: 192.168.42.228
%   R28-2: 192.168.137.228
%   Fallback: 10.238.92.106

    p = inputParser;
    addParameter(p, 'timeout', 3);  % Connection timeout in seconds
    addParameter(p, 'username', 'user');  % SSH username
    addParameter(p, 'verbose', true);
    parse(p, varargin{:});
    
    timeout = p.Results.timeout;
    username = p.Results.username;
    verbose = p.Results.verbose;
    
    % List of Tower IP addresses to try
    tower_ips = {
        '192.168.42.226',   % R26-1
        '192.168.137.226',  % R26-2
        '192.168.42.228',   % R28-1
        '192.168.137.228',  % R28-2
        '10.238.92.106'     % Fallback
    };
    
    tower_names = {'R26-1', 'R26-2', 'R28-1', 'R28-2', 'Fallback'};
    
    success = false;
    tower_ip = '';
    ssh_conn = '';
    
    if verbose
        fprintf('Searching for Tower computer...\n');
    end

    %% Ping all IPs in parallel to find reachable ones quickly
    % Build a single bash command that pings all IPs concurrently
    ping_parts = cell(1, length(tower_ips));
    for i = 1:length(tower_ips)
        ping_parts{i} = sprintf('(ping -c 1 -W %d %s > /dev/null 2>&1 && echo %s) &', ...
            timeout, tower_ips{i}, tower_ips{i});
    end
    parallel_ping_cmd = [strjoin(ping_parts, ' '), ' wait'];
    [~, ping_output] = system(parallel_ping_cmd);

    % Parse which IPs responded
    reachable_ips = strsplit(strtrim(ping_output), newline);
    reachable_ips = reachable_ips(~cellfun('isempty', reachable_ips));

    if isempty(reachable_ips)
        if verbose
            fprintf('  Tower is not reachable.\n');
        end
        return;
    end

    %% SSH-test only reachable IPs (in priority order)
    for i = 1:length(tower_ips)
        ip = tower_ips{i};
        if ~ismember(ip, reachable_ips)
            continue;
        end

        name = tower_names{i};
        if verbose
            fprintf('  Reachable from %s (%s), testing SSH... ', name, ip);
        end

        ssh_test_cmd = sprintf('ssh -o ConnectTimeout=%d -o BatchMode=yes %s@%s "echo ok" 2>/dev/null', ...
            timeout, username, ip);
        [ssh_status, ssh_result] = system(ssh_test_cmd);

        if ssh_status == 0 && contains(ssh_result, 'ok')
            if verbose
                fprintf('SUCCESS!\n');
            end
            success = true;
            tower_ip = ip;
            ssh_conn = sprintf('%s@%s', username, ip);
            return;
        else
            if verbose
                fprintf('SSH failed\n');
            end
        end
    end

    if verbose && ~success
        fprintf('  Could not establish SSH to any reachable Tower address.\n');
    end
end
