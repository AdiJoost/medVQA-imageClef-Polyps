#!/bin/bash

#SBATCH --output="slurm-%j.out"  ## Im Verzeichnis aus dem sbatch aufgerufen wird, wird ein Logfile mit dem Namen slurm-[Jobid].out erstellt.
#SBATCH --error="slurm-%j.err"   ## Ähnlich wie --output. Jedoch ein Log für Fehlermeldungen.
#SBATCH --time=4:30:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein. In diesem Fall ist diese auf 1 Stunde und 30 Minuten gesetzt.
#SBATCH --job-name="test"        ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --cpus-per-task=1        ## Die Anzahl Threads die Slurm starten soll
#SBATCH --ntasks-per-node=64     ## Die Anzahl Prozesse die gestartet werden sollen
#SBATCH --gpus=a100:1		 ## Die Anzahl GPUs (hier eine GPU, mit der Syntax :1)



### Ein etwas anspruchsvolleres Python Script mit dem Namen 'my-script.py' das eine bestehende Conda Umgebung mit dem Namen 'tf' benötigt würde so aussehen
srun --gpus=a100:1 --time=08:00:00 -p students conda run -n medvqa python trainer.py