classdef dbNaverLabs_B1_Test < dbBase
    
    methods
    
        function db= dbNaverLabs_B1_Test()
            db.name= sprintf('NAVERLABS_Indoor_B1_test');
            
            paths= localPaths();
            dbRoot= paths.dsetNAVERLABS;
            db.dbPath= [dbRoot, 'Train/'];
            db.qPath= [dbRoot, 'Test/B1/images/'];
            
            db.dbLoad();
        end
        
    end
    
end

