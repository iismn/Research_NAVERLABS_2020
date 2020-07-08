classdef dbNaverLabs_B1 < dbBase
    
    methods
    
        function db= dbNaverLabs_B1(whichSet)
            % whichSet is one of: train, val
            
            assert( ismember(whichSet, {'train', 'val'}) );
            
            
            
            
            
            
            
            
            db.name= sprintf('NAVERLABS_Indoor_B1_%s', whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetNAVERLABS;
            db.dbPath= [dbRoot, 'images/'];
            db.qPath= [dbRoot, 'images/'];
            
            db.dbLoad();
        end

        
    end
    
end

