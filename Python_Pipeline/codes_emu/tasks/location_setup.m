function params = location_setup(location)
parts = strsplit(location,'-');
prev_dir  = pwd;
if isempty(parts)
    error('location must have the form: PLACE[-OPTIONAL STRINGS]')
end

cd([fileparts(mfilename('fullpath')) filesep 'locations'])

switch parts{1}
    case 'MCW'
        params = MCW_location(location);
    case 'BCM'
        params = BCM_location(location);
    otherwise
        cd(prev_dir)
        error('Location not found')
end

cd(prev_dir)
end

