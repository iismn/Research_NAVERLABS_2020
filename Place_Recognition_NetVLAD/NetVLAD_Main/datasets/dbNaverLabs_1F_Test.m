classdef dbNaverLabs_1F_Test < dbBase
    
    methods
    
        function db= dbNaverLabs_1F_Test()
            db.name= sprintf('NAVERLABS_Indoor_1F_test');
            
            paths= localPaths();
            dbRoot= paths.dsetNAVERLABS;
            db.dbPath= [dbRoot, 'Train/'];
            db.qPath= [dbRoot, 'Test/1F/images/'];
            
            db.dbLoad();
        end
        
    end
    
end

