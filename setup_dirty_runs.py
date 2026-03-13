if __name__ == "__main__":

    
    ggeoms = ["dusty", "shell", "cloudy"]
    geoms = ["h", "c"]
    geomnums = [1.0, 0.01]
    fAs = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ggeom in ggeoms:
        for lgeom, lgeomnum in zip(geoms, geomnums):
            for cfA in fAs:
                dtype = f"fA{cfA:.2f}"
                cfile = f"sge2026_{ggeom}_h_fA1.00.param"

                with open(cfile, 'r') as file:
                    lines_list = [line.rstrip('\n') for line in file]

                taus = [0.001, 0.0025, 0.0050, 0.0075, 0.01,
                        0.025, 0.050, 0.075, 0.10, 0.15, 0.20,
                        0.25, 0.50, 0.75, 1.00, 1.50, 2.00,
                        2.50, 3.0, 3.5, 4.0, 4.5, 5.0,
                        5.5, 6.0, 7.0, 8.0, 9.0, 10.0,
                        15.0, 20.0, 25.0, 30.0, 35.0,
                        40.0, 45.0, 50.0]

                batchlines = []

                for ctau in taus:
                    ofilebase = f"{ggeom}/sge2026_{ggeom}_{lgeom}_{dtype}_tau{ctau:.4f}"
                    batchlines.append(f"./dirty {ofilebase}.param >& {ofilebase}.log")
                    with open(f"{ofilebase}.param", 'w') as file:
                        for cline in lines_list:
                            if "tau=" in cline:
                                cline = f"tau={ctau}"
                            elif "density_ratio=" in cline:
                                cline = "density_ratio={lgeonnum:.2f}"
                            elif "file=" in cline:
                                dprop = f"sge2026_{ggeom}_{lgeom}_fA{cfA:.2f}.param"
                                cline = f"file={dprop}"
                            elif "output_filebase=" in cline:
                                cline = f"output_filebase={ofilebase}"
                            file.write(f"{cline}\n")

                # write batch file
                with open(f"{ggeom}_{lgeom}_{dtype}.batch", "w") as file:
                    for cline in batchlines:
                        file.write(f"{cline}\n")
