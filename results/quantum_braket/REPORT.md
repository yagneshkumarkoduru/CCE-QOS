# CCE-QOS on Amazon Braket

QAOA and analog Ising experiments executed on Amazon Braket devices.
Costs: IonQ Forte Enterprise 1 $0.30/task + $0.08/shot; QuEra Aquila
$0.30/task + $0.01/shot; managed simulators per minute (AWS Pricing API,
us-east-1, read during the experiment session).

| instance | p | device | shots | mean energy | best energy | ratio(mean) | ratio(best) | ground prob | task |
|---|---|---|---|---|---|---|---|---|---|
| chain3 | 1 | local-sim | 4000 | 1.1605 | -0.7000 | 0.6792 | 1.0000 | 0.2880 | - |
| chain3 | 2 | local-sim | 4000 | 0.4326 | -0.7000 | 0.8047 | 1.0000 | 0.3802 | - |
| chain3 | 3 | local-sim | 4000 | -0.3458 | -0.7000 | 0.9389 | 1.0000 | 0.8367 | - |
| random11 | 1 | local-sim | 4000 | -0.0678 | -10.6412 | 0.5365 | 0.9887 | 0.0000 | - |
| random11 | 2 | local-sim | 4000 | -5.9065 | -10.9051 | 0.7862 | 1.0000 | 0.0080 | - |
| random11 | 3 | local-sim | 4000 | -0.3385 | -10.7108 | 0.5481 | 0.9917 | 0.0000 | - |
| chain3 | 1 | sv1 | 4000 | 1.1644 | -0.7000 | 0.6786 | 1.0000 | 0.2920 | ...25af42a06109 |
| chain3 | 2 | sv1 | 4000 | 0.4448 | -0.7000 | 0.8026 | 1.0000 | 0.3698 | ...7e5363e964b5 |
| chain3 | 3 | sv1 | 4000 | -0.3401 | -0.7000 | 0.9379 | 1.0000 | 0.8365 | ...f23383f5da51 |
| random11 | 1 | sv1 | 4000 | -0.1016 | -10.6412 | 0.5380 | 0.9887 | 0.0000 | ...f1eb364f5010 |
| random11 | 2 | sv1 | 4000 | -5.8761 | -10.9051 | 0.7849 | 1.0000 | 0.0075 | ...b92626c8da5a |
| random11 | 3 | sv1 | 4000 | -0.3288 | -10.9051 | 0.5477 | 1.0000 | 0.0003 | ...37f765996315 |

