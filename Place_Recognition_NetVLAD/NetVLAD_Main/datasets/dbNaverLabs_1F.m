classdef dbNaverLabs_1F < dbBase
    
    methods
    
        function db= dbNaverLabs_1F(whichSet)
            % whichSet is one of: train, val
            
            assert( ismember(whichSet, {'train', 'val'}) );
            
            
            
            
            
            
            
            
            db.name= sprintf('NAVERLABS_Indoor_1F_%s', whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetNAVERLABS;
            db.dbPath= [dbRoot, 'images/'];
            db.qPath= [dbRoot, 'images/'];
            
            db.dbLoad();
        end

        
    end
    
end

