% Add folders to path.

addpath(pwd);

cd networks/;
addpath(genpath(pwd));
cd ..;

cd optimizers/;
addpath(genpath(pwd));
cd ..;

cd layers/;
addpath(genpath(pwd));
cd ..;

cd utils/;
addpath(genpath(pwd));
cd ..;

cd plotter/;
addpath(genpath(pwd));
cd ..;


cd datasets/;
addpath(genpath(pwd));
cd ..;


[version, release_date] = simpledeepnettoolbox_version();
fprintf('##########################################################\n');
fprintf('###                                                    ###\n');
fprintf('###          Welcome to SimpleDeepNetToolbox           ###\n');
fprintf('###      (version:%s, released:%s)         ###\n', version, release_date);
fprintf('###                                                    ###\n');
fprintf('##########################################################\n');

