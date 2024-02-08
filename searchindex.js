Search.setIndex({"docnames": ["api", "benchmark", "cite", "docker", "generated/detkit.covariance_matrix", "generated/detkit.design_matrix", "generated/detkit.electrocardiogram", "generated/detkit.get_config", "generated/detkit.get_instructions_per_task", "generated/detkit.logdet", "generated/detkit.loggdet", "generated/detkit.logpdet", "generated/detkit.ortho_complement", "generated/detkit.orthogonalize", "index", "install/compile_source", "install/dependencies", "install/gen_documentation", "install/install", "install/install_wheels", "install/test_package", "install/troubleshooting", "install/virtual_env"], "filenames": ["api.rst", "benchmark.rst", "cite.rst", "docker.rst", "generated/detkit.covariance_matrix.rst", "generated/detkit.design_matrix.rst", "generated/detkit.electrocardiogram.rst", "generated/detkit.get_config.rst", "generated/detkit.get_instructions_per_task.rst", "generated/detkit.logdet.rst", "generated/detkit.loggdet.rst", "generated/detkit.logpdet.rst", "generated/detkit.ortho_complement.rst", "generated/detkit.orthogonalize.rst", "index.rst", "install/compile_source.rst", "install/dependencies.rst", "install/gen_documentation.rst", "install/install.rst", "install/install_wheels.rst", "install/test_package.rst", "install/troubleshooting.rst", "install/virtual_env.rst"], "titles": ["API Reference", "Benchmark Test", "How to Cite", "Using <span class=\"synco\">detkit</span> on Docker", "covariance_matrix", "design_matrix", "electrocardiogram", "get_config", "get_instructions_per_task", "logdet", "loggdet", "logpdet", "ortho_complement", "orthogonalize", "<span class=\"synco\">detkit</span> Documentation", "<span class=\"section-number\">4. </span>Compile from Source", "<span class=\"section-number\">3. </span>Runtime Dependencies", "<span class=\"section-number\">5. </span>Generate Documentation", "Install", "<span class=\"section-number\">1. </span>Install from Wheels", "<span class=\"section-number\">6. </span>Test Package", "<span class=\"section-number\">7. </span>Troubleshooting", "<span class=\"section-number\">2. </span>Install in Virtual Environments"], "terms": {"main": [0, 17, 22], "function": [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 15, 16, 21], "orthogon": [0, 1, 5, 10, 11, 12], "dataset": [0, 4, 5, 6], "util": [0, 1, 3, 15, 16], "detkit": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22], "thi": [1, 3, 6, 8, 9, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22], "page": [1, 2, 14, 17], "demonstr": 1, "matric": [1, 7, 8, 10, 14, 15], "from": [1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22], "real": [1, 14], "applic": [1, 2, 10, 11, 14], "we": [1, 8, 14, 22], "reader": 1, "1": [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "detail": [1, 19], "numer": [1, 14], "here": 1, "provid": [1, 8, 19], "supplement": 1, "how": 1, "reproduc": 1, "aim": [1, 8], "comput": [1, 2, 4, 7, 8, 9, 10, 11, 14, 15, 16], "begin": [1, 5], "align": 1, "mathrm": [1, 9, 10, 11], "logdet": [1, 10, 11, 21], "mathbf": [1, 4, 5, 9, 10, 11, 12, 13], "A": [1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15], "x": [1, 5, 8, 10, 11, 12, 13], "interc": [1, 10, 11, 12, 13], "tag": 1, "ld1": 1, "n": [1, 4, 5, 8, 9, 10, 11, 12], "ld2": 1, "u": [1, 10, 11, 16], "_": [1, 12], "mathcal": 1, "perp": [1, 12], "ld3": 1, "end": [1, 4, 5, 6, 16], "where": [1, 4, 5, 7, 10, 11, 12, 13, 18], "mathbb": [1, 4], "r": [1, 10, 11, 16, 17, 20], "time": [1, 4, 6, 7, 10, 11, 12, 17, 18], "p": [1, 3, 12], "ar": [1, 3, 6, 7, 8, 10, 11, 14, 15, 16, 18, 19, 22], "given": [1, 4, 6, 9, 10, 11, 12], "i": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22], "m": [1, 5, 8, 10, 11, 15, 17, 19, 20, 22], "orthonorm": [1, 5, 10, 11, 12], "basi": 1, "imag": [1, 14, 21], "The": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22], "section": [1, 15, 18], "3": [1, 3, 6, 8, 12, 13, 14], "motiv": 1, "definit": [1, 7, 9, 10, 11], "In": [1, 15, 16, 22], "three": 1, "relat": [1, 8], "abov": [1, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 20, 21, 22], "implement": [1, 9, 14], "loggdet": [1, 8, 9, 11], "accept": [1, 10], "paramet": [1, 4, 5, 6, 8, 9, 10, 11, 12, 13], "method": [1, 10, 11, 12, 13, 21, 22], "possibl": 1, "valu": [1, 10, 11], "legaci": [1, 10, 11], "proj": [1, 10, 11], "comp": [1, 10, 11], "correspond": [1, 6], "respect": 1, "compar": 1, "measur": [1, 6, 8, 14], "process": [1, 2, 4, 5, 8, 9, 10, 11, 14, 15, 21], "empir": 1, "flop": [1, 8, 10, 11, 14, 16], "consid": [1, 2, 14], "four": 1, "case": [1, 5, 10, 11, 18], "symmetr": [1, 7, 9, 10, 11], "posit": [1, 9, 10, 11], "spd": [1, 9, 10, 11], "complex": [1, 6, 8], "us": [1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22], "either": [1, 8, 10, 11, 15, 16, 18, 19, 20, 22], "shown": [1, 4, 6, 14, 15, 20], "figur": 1, "below": [1, 14, 16], "see": [1, 3, 10, 11, 14, 15, 16, 17, 21], "also": [1, 3, 6, 8, 10, 11, 14, 16], "our": 1, "experi": 1, "matrix": [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "gener": [1, 5, 8, 9, 10, 11, 14, 15, 16, 18], "trigonometr": 1, "domain": 1, "t": [1, 4, 9, 10, 11, 12, 13], "0": [1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15], "x_": [1, 5], "ij": [1, 4, 5], "j": [1, 2, 4, 5, 14], "sin": [1, 5], "t_i": [1, 5], "pi": [1, 5], "2k": [1, 5], "co": [1, 5], "frac": [1, 4, 5, 8], "number": [1, 5, 8, 12], "row": [1, 5, 12], "fix": 1, "size": [1, 4, 8, 12, 15], "2": [1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 21], "9": [1, 3, 4, 5, 14, 15, 16, 17], "vari": [1, 8], "p_1": 1, "dot": 1, "p_l": 1, "p_j": 1, "jn": 1, "l": [1, 3, 6], "30": [1, 4], "can": [1, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22], "design_matrix": [1, 4], "column": [1, 5, 10, 11, 12, 13], "code": [1, 8, 14, 15, 17, 18, 20], "produc": [1, 21], "first": [1, 17, 19], "instal": [1, 10, 11, 16, 17, 20, 21], "pip": [1, 14, 15, 17, 18, 20, 22], "import": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 21], "ortho": [1, 5], "true": [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 21], "an": [1, 6, 8, 10, 11, 14, 21, 22], "covariance_matrix": [1, 5, 6], "which": [1, 10, 11, 15, 16, 20, 22], "covari": [1, 4, 6], "base": [1, 4, 15, 18], "autocovari": [1, 4], "electrocardiogram": [1, 4], "ecg": [1, 4, 6], "signal": [1, 4, 6], "describ": [1, 3, 14], "obtain": [1, 8, 16], "segment": [1, 21], "To": [1, 3, 7, 8, 11, 15, 16, 19, 20, 21, 22], "stationari": [1, 4], "remov": [1, 6, 21], "baselin": [1, 6], "wander": [1, 6], "variat": 1, "orang": 1, "curv": 1, "pass": [1, 6], "move": [1, 6], "averag": [1, 6], "filter": [1, 6], "5": [1, 6, 8, 15], "second": [1, 4, 6, 12], "window": [1, 14, 15, 16, 17], "length": [1, 6], "reduc": 1, "nois": [1, 6, 10, 11], "low": [1, 6], "cut": [1, 6], "frequenc": [1, 4, 6], "45": [1, 6], "hz": [1, 4, 6], "smooth": 1, "start": [1, 3, 4, 6], "10": [1, 2, 6, 14], "bw_window": [1, 6], "freq_cut": [1, 6], "plot_bw": [1, 6], "follow": [1, 2, 3, 5, 7, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21, 22], "It": [1, 4, 8, 9, 16, 21], "assum": [1, 4, 8, 9, 10, 11, 22], "wide": [1, 4], "sens": [1, 4], "stochast": [1, 4], "so": [1, 4, 6, 12, 13, 19, 21], "its": [1, 4, 6, 8, 14, 15, 18, 19], "kappa": [1, 4], "delta": [1, 4], "e": [1, 4, 9, 10, 11, 15, 16, 17, 21], "f": [1, 4], "bar": [1, 4], "lag": [1, 4], "expect": [1, 4], "oper": [1, 3, 4, 7, 8, 10, 11, 15, 16, 19], "mean": [1, 4, 15, 21], "boldsymbol": [1, 4], "defin": [1, 4, 8, 9, 10, 11], "compon": [1, 4, 6], "a_": 1, "vert": [1, 4], "f_": [1, 4], "nu": [1, 4], "360": [1, 4, 6], "sampl": [1, 4, 6], "autocorrel": [1, 4, 6], "note": [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], "toeplitz": 1, "tau": [1, 4], "k": [1, 4, 6], "sigma": [1, 4], "varianc": [1, 4], "b": [1, 9, 10, 11], "show": [1, 7, 22], "correl": [1, 4, 6], "c": [1, 2, 3, 6, 8, 9, 10, 11, 14, 16, 17, 18, 19, 22], "eigenvalu": [1, 6], "indic": [1, 21], "henc": 1, "all": [1, 3, 14, 20, 22], "cor": [1, 4], "ecg_start": [1, 4], "ecg_end": [1, 4], "ecg_wrap": [1, 4], "independ": [1, 10, 11], "dure": [1, 7, 9, 10, 11, 14, 16], "user": [1, 3, 10, 11, 15, 16, 18], "randomli": 1, "consist": 1, "two": [1, 6, 22], "differ": [1, 8, 12], "gramian": [1, 7, 8, 15], "multipl": [1, 4, 7, 8, 15, 21], "within": 1, "sourc": [1, 3, 14, 16, 17, 18, 20, 22], "packag": [1, 2, 3, 7, 8, 10, 11, 14, 15, 16, 18, 19, 21, 22], "do": [1, 14, 15, 19, 20, 21, 22], "need": [1, 14, 15, 16, 18, 22], "compil": [1, 7, 8, 11, 14, 16, 18], "both": [1, 4, 6, 11, 19], "git": [1, 15, 17, 20], "clone": [1, 15, 17, 20], "http": [1, 2, 3, 14, 15, 17, 20, 21], "www": 1, "github": [1, 3, 14, 15, 17, 20], "com": [1, 3, 15, 17, 20], "am": [1, 2, 10, 11, 14, 15, 17, 19, 20, 22], "modifi": [1, 7, 14, 15], "_definit": [1, 7], "h": [1, 7, 15, 16], "use_symmetri": [1, 7, 15], "enabl": [1, 3, 7, 15, 16], "gamma": 1, "4": [1, 3, 12], "disabl": [1, 21], "For": [1, 4, 14, 15, 16, 18, 19, 21, 22], "each": [1, 15, 16], "next": [1, 19, 21, 22], "except": 1, "purpos": 1, "requir": [1, 17, 18, 20, 21], "through": [1, 14, 15], "conda": [1, 14, 15, 18, 21], "pre": [1, 3, 14], "make": [1, 10, 11, 15, 17, 20], "sure": [1, 10, 11, 15, 17], "linux": [1, 3, 10, 11, 14, 15, 16], "perf": [1, 10, 11, 18], "tool": [1, 10, 11, 15, 18], "runtim": [1, 3, 7, 10, 11, 14, 18, 21], "depend": [1, 8, 14, 18, 19, 22], "onli": [1, 3, 6, 10, 11, 15, 16, 21], "system": [1, 3, 8, 10, 11, 15, 16, 19, 21], "due": 1, "script": [1, 20], "py": [1, 15, 17], "locat": [1, 20], "directori": [1, 3, 4, 6, 15, 17, 20], "usag": [1, 18], "list": [1, 3, 11, 22], "argument": [1, 4, 16], "command": [1, 3, 10, 11, 15, 16, 17, 19, 20, 21, 22], "line": [1, 6], "call": 1, "help": 1, "option": [1, 3, 6, 10, 11, 18, 21], "cd": [1, 15, 17, 20], "python": [1, 2, 3, 9, 14, 15, 17, 18, 19, 20, 22], "print": [1, 3, 4, 6, 8, 17], "int": [1, 4, 5, 8, 9, 10, 11], "log2": 1, "power": [1, 6], "func": 1, "str": [1, 6], "type": [1, 8, 9, 10, 11], "logpdet": [1, 8, 9, 10], "bla": [1, 7, 10, 11], "exist": [1, 22], "librari": [1, 3, 7, 10, 11, 14, 15, 21], "numpi": [1, 4, 5, 6, 8, 9, 10, 11, 12, 13], "scipi": [1, 10, 11], "otherwis": [1, 15], "cython": [1, 14, 15], "repeat": 1, "default": [1, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 22], "num": 1, "ratio": 1, "50": [1, 6], "v": [1, 3], "verbos": [1, 4, 6], "fals": [1, 4, 5, 6, 7, 9, 10, 11, 12], "messag": [1, 21], "exampl": [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20], "8": [1, 5, 14], "256": [1, 5], "arrai": [1, 5, 6, 12, 13], "linspac": 1, "512": [1, 4, 5], "100": 1, "ten": 1, "point": [1, 8], "cdot": 1, "job": 1, "jobfil": 1, "one": [1, 10, 11, 22], "processor": [1, 8, 10, 11, 15], "thread": 1, "accur": [1, 14], "torqu": 1, "workload": 1, "manag": [1, 3, 15], "submit": 1, "jobfile_benchmark": 1, "pb": 1, "qsub": 1, "slurm": 1, "sh": [1, 10, 11, 16, 22], "sbatch": 1, "instanc": [1, 4, 7, 8, 15, 17, 22], "num_ratio": 1, "store": 1, "pickle_result": 1, "ha": [1, 3, 15], "name": [1, 8, 22], "filenam": [1, 4, 6], "benchmark_loggdet_9_gram": 1, "pickl": 1, "symmetri": [1, 15], "benchmark_loggdet_9_no": 1, "gram": [1, 5, 7, 12, 13], "without": [1, 3, 15, 16], "notebook": [1, 14], "benchmark_plot": 1, "ipynb": 1, "svg": [1, 4, 6], "pdf": [1, 2, 4, 6, 14], "These": [1, 19], "normal": 1, "recal": 1, "Such": 1, "advantag": [1, 10, 11], "scale": 1, "remain": 1, "unchang": 1, "": [1, 2, 3, 10, 11, 14, 15, 16, 17, 19, 22], "shadden": [1, 2, 10, 11, 14], "2023": [1, 2, 14], "singular": [1, 2, 9, 10, 11, 14], "woodburi": [1, 2, 10, 11, 14], "pseudo": [1, 2, 8, 9, 10, 11, 14], "determin": [1, 2, 7, 8, 9, 10, 11, 12, 14], "ident": [1, 2, 10, 11, 13, 14], "gaussian": [1, 2, 8, 9, 10, 11, 14], "regress": [1, 2, 8, 9, 10, 11, 14], "appli": [1, 2, 6, 10, 11, 14], "mathemat": [1, 2, 8, 14], "452": [1, 2, 14], "128032": [1, 2, 14], "doi": [1, 2, 6, 14], "bibtex": [1, 2, 14], "articl": [1, 2, 14], "amc": [1, 2, 14], "titl": [1, 2, 14], "journal": [1, 2, 14], "volum": [1, 2, 14], "year": [1, 2, 14], "issn": [1, 2, 14], "0096": [1, 2, 14], "3003": [1, 2, 14], "org": [1, 2, 14, 21], "1016": [1, 2, 14], "author": [1, 2, 14], "siavash": [1, 2, 14], "shawn": [1, 2, 14], "moodi": [1, 6], "gb": [1, 6], "mark": [1, 6, 22], "rg": [1, 6], "impact": [1, 6], "mit": [1, 6], "bih": [1, 6], "arrhythmia": [1, 6], "databas": [1, 6], "ieee": [1, 6], "eng": [1, 6], "med": [1, 6], "biol": [1, 6], "20": [1, 6], "mai": [1, 6, 8, 10, 11, 15, 16, 18, 19, 21, 22], "june": [1, 6], "2001": [1, 6], "pmid": [1, 6], "11446209": [1, 6], "13026": [1, 6], "c2f305": [1, 6], "goldberg": [1, 6], "al": [1, 6], "amar": [1, 6], "lan": [1, 6], "glass": [1, 6], "hausdorff": [1, 6], "jm": [1, 6], "ivanov": [1, 6], "pch": [1, 6], "mietu": [1, 6], "je": [1, 6], "peng": [1, 6], "stanlei": [1, 6], "he": [1, 6], "physiobank": [1, 6], "physiotoolkit": [1, 6], "physionet": [1, 6], "new": [1, 6, 22], "research": [1, 6], "resourc": [1, 6, 18], "physiolog": [1, 6], "circul": [1, 6], "101": [1, 6], "23": [1, 6], "e215": [1, 6], "e220": [1, 6], "1161": [1, 6], "01": [1, 6], "cir": [1, 6], "If": [2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 19, 21, 22], "you": [2, 3, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22], "publish": [2, 14], "work": [2, 10, 11, 14, 15, 16], "pleas": [2, 14, 21], "manuscript": [2, 14], "2022": [2, 10, 11, 14], "misc": [2, 14], "zenodo": [2, 14], "6395320": [2, 14], "howpublish": [2, 14], "url": [2, 14], "pypi": [2, 14, 19], "project": [2, 3], "instruct": [3, 7, 8, 10, 11, 15, 16, 18, 21], "other": [3, 14], "engin": 3, "document": [3, 15, 18, 19], "ubuntu": [3, 10, 11, 15, 16, 17], "debian": [3, 15, 16, 17], "sudo": [3, 10, 11, 15, 16, 17, 22], "apt": [3, 10, 11, 15, 16, 17], "updat": 3, "ca": 3, "certif": 3, "curl": 3, "gnupg": 3, "lsb": 3, "releas": [3, 15], "mkdir": 3, "etc": [3, 22], "keyr": 3, "fssl": 3, "download": 3, "gpg": 3, "dearmor": 3, "o": [3, 12, 17], "echo": [3, 10, 11, 16], "quot": [3, 15, 17], "deb": 3, "arch": [3, 14], "dpkg": 3, "architectur": [3, 14], "sign": [3, 8, 9, 10, 11], "lsb_releas": 3, "stabl": 3, "tee": 3, "d": [3, 22], "gt": [3, 10, 11, 16], "dev": [3, 10, 11, 15, 16], "null": [3, 10, 11, 16], "ce": 3, "cli": 3, "containerd": 3, "io": 3, "compos": 3, "plugin": 3, "cento": [3, 15, 16, 17], "7": [3, 15, 16, 17], "yum": [3, 15, 16, 17], "y": [3, 15, 16, 17, 19, 22], "config": [3, 7, 15], "add": [3, 7, 8, 15], "repo": 3, "systemctl": 3, "servic": [3, 21], "rhel": [3, 15, 16, 17], "configur": [3, 7, 8, 10, 11, 16, 17, 18], "password": 3, "groupadd": 3, "usermod": 3, "ag": 3, "Then": [3, 15, 22], "log": [3, 8, 9, 10, 11], "out": 3, "back": 3, "virtual": [3, 14, 18, 20], "machin": [3, 14], "restart": 3, "chang": [3, 15, 22], "take": [3, 8, 10, 11], "effect": [3, 6, 8], "pull": [3, 14], "same": [3, 7, 8, 12, 14, 15], "cuda": [3, 14], "usr": [3, 15, 16], "local": [3, 15, 16], "bin": [3, 15, 22], "python3": 3, "interpret": 3, "ipython": 3, "jupyt": [3, 14], "editor": 3, "vim": 3, "some": [3, 14], "variou": 3, "check": [3, 5, 12, 13, 15], "host": 3, "driver": [3, 14], "version": [3, 14, 15, 16, 19], "avail": [3, 10, 11, 14, 15, 16, 18, 19], "devic": [3, 6], "nvida": 3, "smi": 3, "open": [3, 16], "directli": [3, 8, 10, 11, 15], "startup": 3, "automat": [3, 20], "bash": 3, "shell": 3, "entrypoint": 3, "mount": 3, "home": 3, "onto": 3, "root": [3, 15, 22], "access": [3, 15], "insid": [3, 15], "should": [3, 10, 11, 15, 16, 17, 21, 22], "repositori": [3, 15, 17, 20], "distribut": 3, "id": 3, "version_id": 3, "gpgkei": 3, "kei": 3, "dnf": [3, 15, 16, 17], "contan": 3, "x86_64": 3, "el7": 3, "rpm": 3, "simpli": 3, "ani": [3, 15, 17, 18, 21, 22], "earlier": 3, "plot": [4, 6], "creat": [4, 5, 12, 13, 22], "pace": 4, "bool": [4, 5, 6, 9, 10, 11, 12], "instead": [4, 8, 19, 22], "return": [4, 5, 6, 7, 8, 9, 10, 11, 15], "float": [4, 6, 8, 9, 10, 11], "wrap": [4, 14], "string": [4, 6], "rather": [4, 6], "save": [4, 6], "doe": [4, 6, 8, 9, 10, 11], "contain": [4, 6, 14], "file": [4, 6, 7, 15], "extens": [4, 6], "format": [4, 6], "have": [4, 6, 8, 10, 11, 12, 14, 15, 16, 19, 21], "path": [4, 6, 17], "current": [4, 6, 8, 15, 22], "ndarrai": [4, 5], "sigma_": 4, "specifi": 4, "total": 4, "span": 4, "84": 4, "versu": 4, "scalar": 4, "detit": 4, "num_row": 5, "num_col": 5, "design": 5, "2d": 5, "schmidt": [5, 12, 13], "04419417": 5, "09094864": 5, "06243905": 5, "09532571": 5, "09006862": 5, "06243787": 5, "09299386": 5, "08918863": 5, "06243433": 5, "09066257": 5, "load": [6, 15, 18], "1d": 6, "bw": 6, "zero": [6, 10, 11, 12, 16], "set": [6, 10, 11, 15, 17, 19, 21], "limit": 6, "inf": [6, 9], "perform": [6, 7, 10, 11, 14, 15, 16, 21], "origin": 6, "along": [6, 19], "axi": 6, "trend": 6, "caus": [6, 21], "respir": 6, "movement": 6, "person": 6, "usual": 6, "rang": 6, "unfortun": 6, "high": [6, 14], "cleanli": 6, "best": [6, 21], "approach": 6, "far": 6, "wa": [6, 7], "kernel": [6, 10, 11, 16, 21], "durat": 6, "about": 6, "critic": 6, "60": 6, "interfer": 6, "refer": [6, 10, 11, 14, 16, 19, 21], "dict": 7, "dictionari": 7, "field": 7, "boolean": 7, "whether": [7, 9, 10, 11, 12], "when": [7, 8, 9, 10, 11, 18, 21], "full": [7, 10, 11, 16], "use_openmp": [7, 11, 15], "openmp": [7, 18, 21], "parallel": [7, 10, 11, 15], "count_perf": [7, 15], "count": [7, 8, 10, 11, 14, 15, 16], "hardwar": [7, 8, 10, 11], "chunk_task": [7, 15], "chunk": [7, 15], "multipli": [7, 8], "ma": 7, "similar": [7, 15], "techniqu": 7, "everi": 7, "five": 7, "get_instructions_per_task": [7, 10, 11, 16], "recompil": 7, "support": [7, 8, 10, 11, 16], "share": [7, 15], "memori": [7, 9, 15], "improv": 7, "task": 8, "matmat": 8, "dtype": 8, "float64": [8, 9, 10, 11], "singl": [8, 21], "benchmark": 8, "goal": 8, "algorithm": [8, 9, 10, 11, 14], "fuse": 8, "cannot": [8, 10, 11, 14, 15, 18], "divid": 8, "run": [8, 15, 16, 20, 21], "desir": [8, 15], "anoth": 8, "unit": 8, "basic": 8, "lu": 8, "factor": [8, 10, 11], "choleski": [8, 9, 10, 11], "decomposit": [8, 9, 10, 11], "lup": [8, 10, 11], "tend": 8, "infin": 8, "float32": [8, 9, 10, 11], "float128": [8, 9, 10, 11], "test": [8, 10, 11, 14, 16, 18, 22], "data": 8, "inst": 8, "featur": 8, "term": [8, 9, 11], "precis": [8, 9, 10], "result": [8, 21], "converg": 8, "veri": 8, "larg": 8, "uniqu": 8, "asymptot": 8, "c_": 8, "infti": 8, "sever": [8, 20], "find": [8, 21], "appear": 8, "choic": 8, "get_instructions_per_count": 8, "random": [8, 9, 10, 11, 12, 13, 14], "1000": [8, 9, 10, 11], "500": [8, 10, 11], "rng": [8, 9, 10, 11], "randomst": [8, 9, 10, 11], "rand": [8, 9, 10, 11, 12, 13], "loggdet_": 8, "11009228170": 8, "benchmark_inst": 8, "benchmark_int": 8, "21": 8, "calcul": 8, "2110324959": 8, "coeffici": 8, "11": [8, 14, 21], "sym_po": [9, 10, 11], "overwrite_a": 9, "log_": [9, 10, 11], "det": [9, 10, 11], "array_lik": [9, 10, 11, 12, 13], "squar": 9, "int32": [9, 10, 11], "int64": [9, 10, 11], "cast": [9, 10, 11], "twice": [9, 10, 11], "fast": [9, 10, 11], "verifi": [9, 10, 11], "input": [9, 12, 13], "overwritten": [9, 12, 13], "less": 9, "could": 9, "potenti": 9, "slightli": 9, "faster": [9, 10, 11], "neg": [9, 10, 11], "rais": [9, 10, 11, 14], "runtimeerror": [9, 10, 11], "error": [9, 10, 11, 21], "plu": [9, 10, 11], "wrapper": [9, 10, 11], "1710": [9, 10], "9576831500378": [9, 10], "3421": [9, 10], "9153663693114": [9, 10], "xp": [10, 11, 12], "none": [10, 11], "x_orth": [10, 11, 12], "use_scipi": [10, 11], "gdet": [10, 11], "invert": [10, 11], "rank": [10, 11], "rectangular": [10, 11], "complement": [10, 11], "equat": [10, 11], "bott": [10, 11], "duffin": [10, 11], "invers": [10, 11], "compress": [10, 11], "retir": [10, 11], "newer": [10, 11], "model": [10, 11], "intel": [10, 11, 15, 16], "cpu": [10, 11], "relev": [10, 11], "around": [10, 11, 12, 13], "fortran": [10, 11], "routin": [10, 11], "lapack": [10, 11], "develop": [10, 11, 15], "degener": [10, 11], "permiss": [10, 11, 16], "counter": [10, 11, 16], "valueerror": [10, 11], "incorrect": [10, 11, 21], "howev": [10, 11, 15, 21], "altern": [10, 11, 15, 16, 19, 22], "formul": [10, 11], "thu": [10, 11], "inner": [10, 11], "spars": 10, "likelihood": [10, 11], "On": [10, 11], "get": [10, 11, 14, 15, 16, 17], "common": [10, 11, 14, 16, 22], "unam": [10, 11, 16], "39": [10, 11, 16], "proc": [10, 11, 16], "sy": [10, 11, 16, 17], "perf_event_paranoid": [10, 11, 16], "stat": [10, 11, 16], "dd": [10, 11, 16], "properli": [10, 11, 15, 16, 20], "output": [10, 11, 12, 13, 15, 16], "empti": [10, 11, 16], "estim": [10, 11], "fail": [10, 11], "sinc": [10, 11, 15, 21], "recent": [10, 11], "pdet": 11, "enhanc": 11, "variabl": [11, 17, 18, 21], "get_config": 11, "680": 11, "9183141420649": 11, "2059": 11, "6208046883685": 11, "8520537034": 11, "against": 12, "place": [12, 13], "must": 12, "thei": 12, "alreadi": [12, 17, 21], "q": 12, "satisfi": [12, 13], "inplac": [12, 13], "Not": 12, "seed": [12, 13], "6": [12, 13, 21], "decim": [12, 13], "806": 12, "207": 12, "017": 12, "91": 12, "93": 12, "903": 12, "455": 12, "again": [12, 13], "14": 12, "492": 12, "345": 12, "283": 12, "892": 12, "336": 12, "013": 12, "237": 12, "222": 12, "15": [12, 13, 15, 16, 21], "ortho_compl": 13, "267": 13, "845": 13, "42": 13, "97": 13, "065": 13, "687": 13, "involv": 14, "learn": 14, "anaconda": [14, 19, 22], "cloud": [14, 19], "hub": 14, "success": 14, "been": [14, 21], "tabl": 14, "continu": [14, 21], "integr": 14, "12": 14, "x86": 14, "64": 14, "aarch": 14, "maco": [14, 15, 16, 17, 21], "arm": 14, "wheel": [14, 15, 18], "issu": [14, 18], "build": [14, 15, 17], "exclus": 14, "complet": [14, 15], "guid": 14, "environ": [14, 17, 18, 20, 21], "troubleshoot": [14, 18], "come": [14, 15], "nvidia": 14, "graphic": 14, "compat": 14, "toolkit": [14, 15], "deploi": 14, "gpu": 14, "api": [14, 16], "launch": 14, "onlin": 14, "interact": 14, "evalu": 14, "novel": 14, "underli": 14, "execut": [14, 17, 21], "welcom": 14, "via": [14, 16, 19], "request": 14, "feel": 14, "comfort": 14, "bug": 14, "report": 14, "scalabl": 14, "want": [15, 22], "debug": 15, "mode": 15, "walk": 15, "gcc": [15, 16], "clang": [15, 16], "llvm": [15, 16, 21], "unix": [15, 17], "microsoft": [15, 16], "visual": [15, 16], "studio": 15, "msvc": 15, "gnu": 15, "essenti": 15, "group": [15, 16], "brew": [15, 16, 17], "libomp": [15, 16, 18], "export": [15, 17], "cxx": 15, "cc": 15, "g": [15, 21], "llvn": 15, "extra": 15, "makecach": 15, "oneapi": 15, "libgomp1": [15, 16], "libgomp": [15, 16], "homebrew": [15, 16], "keg": [15, 16], "establish": [15, 16], "symbol": [15, 16], "link": [15, 16, 21], "libomp_dir": [15, 16], "prefix": [15, 16], "ln": [15, 16], "sf": [15, 16], "includ": [15, 16], "omp": [15, 16, 21], "ompt": [15, 16], "lib": [15, 16], "dylib": [15, 16, 21], "cython_build_in_sourc": 15, "By": [15, 16, 21], "outsid": 15, "directri": 15, "powershel": [15, 17], "env": [15, 17, 22], "later": 15, "clean": [15, 17], "them": 15, "cython_build_for_doc": [15, 17], "built": [15, 17], "suitabl": [15, 21], "product": 15, "slower": 15, "debug_mod": 15, "gdb": 15, "With": 15, "larger": 15, "long_int": 15, "long": 15, "integ": 15, "unsigned_long_int": 15, "unsign": 15, "vector": 15, "functionalit": 15, "perf_tool": 15, "peroform": 15, "conseqqut": 15, "addit": 15, "gamian": 15, "privileg": 15, "setup": [15, 17], "onc": [15, 19], "your": [15, 18, 19, 21, 22], "alwai": 15, "somewher": 15, "els": 15, "among": 16, "while": 16, "rest": 16, "typic": 16, "most": 16, "well": 16, "crucial": 16, "part": 16, "appl": 16, "xcode": 16, "even": 16, "readili": 16, "still": 16, "separ": 16, "choos": 16, "explicitli": 16, "specif": [16, 18, 21], "known": 16, "grant": 16, "abl": [16, 21], "100000": 16, "befor": 17, "had": 17, "previou": 17, "especi": 17, "pandoc": 17, "scoop": 17, "doc": 17, "txt": [17, 20], "python_path": 17, "dirnam": 17, "now": 17, "html": 17, "bat": 17, "found": [17, 18, 21], "index": 17, "major": 18, "easili": 18, "more": [18, 21], "inform": [18, 21], "addition": 18, "explor": 18, "might": 18, "necessari": 18, "advanc": 18, "virtualenv": 18, "sphinx": 18, "pytest": 18, "tox": 18, "initi": [18, 22], "offer": 19, "varieti": 19, "conveni": 19, "wai": 19, "ensur": [19, 21], "ensurepip": 19, "upgrad": 19, "further": 19, "up": 19, "readi": 19, "forg": 19, "sub": 20, "mv": 20, "renam": 20, "upon": 21, "encount": 21, "importerror": 21, "dlopen": 21, "site": [21, 22], "_function": 21, "cpython": 21, "311": 21, "darwin": 21, "loader_path": 21, "referenc": 21, "imat": [21, 22], "reason": 21, "did": 21, "0x80000034": 21, "unknown": 21, "resolv": 21, "slq": 21, "libiomp5": 21, "hint": 21, "copi": 21, "program": 21, "That": 21, "danger": 21, "degrad": 21, "thing": 21, "avoid": 21, "static": 21, "As": 21, "unsaf": 21, "unsupport": 21, "undocu": 21, "workaround": 21, "kmp_duplicate_lib_ok": 21, "allow": 21, "crash": 21, "silent": 21, "abort": 21, "trap": 21, "associ": 21, "challeng": 21, "handl": 21, "itself": 21, "attempt": 21, "suggest": 21, "lead": 21, "fault": 21, "solut": 21, "step": [21, 22], "previous": 21, "unset": 21, "math": 21, "nomkl": 21, "mkl": 21, "mention": 21, "occupi": 22, "clutter": 22, "isol": 22, "give": 22, "imate_env": 22, "activ": 22, "exit": 22, "deactiv": 22, "miniconda": 22, "init": 22, "close": 22, "reopen": 22, "termin": 22, "after": 22, "info": 22, "profil": 22, "asterisk": 22, "stage": 22}, "objects": {"detkit": [[4, 0, 1, "", "covariance_matrix"], [5, 0, 1, "", "design_matrix"], [6, 0, 1, "", "electrocardiogram"], [7, 0, 1, "", "get_config"], [8, 0, 1, "", "get_instructions_per_task"], [9, 0, 1, "", "logdet"], [10, 0, 1, "", "loggdet"], [11, 0, 1, "", "logpdet"], [12, 0, 1, "", "ortho_complement"], [13, 0, 1, "", "orthogonalize"]]}, "objtypes": {"0": "py:function"}, "objnames": {"0": ["py", "function", "Python function"]}, "titleterms": {"api": 0, "refer": [0, 1], "benchmark": [1, 14], "test": [1, 20], "descript": 1, "dataset": 1, "configur": [1, 15], "set": 1, "perform": 1, "run": [1, 3], "cluster": 1, "output": 1, "file": 1, "plot": 1, "result": 1, "how": [2, 14], "cite": [2, 14], "us": 3, "detkit": [3, 14, 15], "docker": [3, 14], "content": 3, "instal": [3, 14, 15, 18, 19, 22], "get": 3, "imag": 3, "exampl": 3, "contain": 3, "deploi": 3, "gpu": 3, "nvidia": 3, "toolkit": 3, "covariance_matrix": 4, "design_matrix": 5, "electrocardiogram": 6, "get_config": 7, "get_instructions_per_task": 8, "logdet": 9, "loggdet": 10, "logpdet": 11, "ortho_compl": 12, "orthogon": 13, "document": [14, 17], "support": 14, "platform": 14, "list": 14, "function": 14, "tutori": 14, "featur": 14, "contribut": 14, "relat": 14, "project": 14, "compil": [15, 17], "from": [15, 19], "sourc": 15, "when": 15, "c": 15, "requir": [15, 16], "openmp": [15, 16], "time": 15, "environ": [15, 22], "variabl": 15, "option": [15, 16], "runtim": 16, "depend": 16, "perf": 16, "tool": 16, "gener": 17, "packag": [17, 20], "sphinx": 17, "wheel": 19, "pip": 19, "conda": [19, 22], "pytest": 20, "tox": 20, "troubleshoot": 21, "cannot": 21, "load": 21, "libomp": 21, "issu": 21, "initi": 21, "virtual": 22, "virtualenv": 22}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"API Reference": [[0, "api-reference"]], "Benchmark Test": [[1, "benchmark-test"]], "Test Description": [[1, "test-description"]], "Dataset": [[1, "dataset"]], "Configure Settings": [[1, "configure-settings"]], "Perform Test": [[1, "perform-test"]], "Run on Cluster": [[1, "run-on-cluster"]], "Output Files": [[1, "output-files"]], "Plot Results": [[1, "plot-results"]], "References": [[1, "references"]], "How to Cite": [[2, "how-to-cite"], [14, "how-to-cite"]], "Using detkit on Docker": [[3, "using-project-on-docker"]], "Contents": [[3, "contents"]], "Install Docker": [[3, "install-docker"]], "Get detkit Docker Image": [[3, "get-project-docker-image"]], "Examples of Using detkit Docker Container": [[3, "examples-of-using-project-docker-container"]], "Deploy detkit Docker Container on GPU": [[3, "deploy-project-docker-container-on-gpu"]], "Install NVIDIA Container Toolkit": [[3, "install-nvidia-container-toolkit"]], "Run detkit Docker Container on GPU": [[3, "run-project-docker-container-on-gpu"]], "covariance_matrix": [[4, "covariance-matrix"]], "design_matrix": [[5, "design-matrix"]], "electrocardiogram": [[6, "electrocardiogram"]], "get_config": [[7, "get-config"]], "get_instructions_per_task": [[8, "get-instructions-per-task"]], "logdet": [[9, "logdet"]], "loggdet": [[10, "loggdet"]], "logpdet": [[11, "logpdet"]], "ortho_complement": [[12, "ortho-complement"]], "orthogonalize": [[13, "orthogonalize"]], "detkit Documentation": [[14, "project-documentation"]], "Supported Platforms": [[14, "supported-platforms"]], "Install": [[14, "install"], [18, "install"]], "Docker": [[14, "docker"]], "List of Functions": [[14, "list-of-functions"]], "Tutorials": [[14, "tutorials"]], "Benchmarks": [[14, "benchmarks"]], "Features": [[14, "features"]], "How to Contribute": [[14, "how-to-contribute"]], "Related Projects": [[14, "related-projects"]], "Compile from Source": [[15, "compile-from-source"]], "When to Compile detkit": [[15, "when-to-compile-project"]], "Install C++ Compiler (Required)": [[15, "install-c-compiler-required"]], "Install OpenMP (Required)": [[15, "install-openmp-required"]], "Configure Compile-Time Environment Variables (Optional)": [[15, "configure-compile-time-environment-variables-optional"]], "Compile and Install": [[15, "compile-and-install"]], "Runtime Dependencies": [[16, "runtime-dependencies"]], "OpenMP (Required)": [[16, "openmp-required"]], "Perf Tool (Optional)": [[16, "perf-tool-optional"]], "Generate Documentation": [[17, "generate-documentation"]], "Compile Package": [[17, "compile-package"]], "Generate Sphinx Documentation": [[17, "generate-sphinx-documentation"]], "Install from Wheels": [[19, "install-from-wheels"]], "Install with pip": [[19, "install-with-pip"]], "Install with conda": [[19, "install-with-conda"]], "Test Package": [[20, "test-package"]], "Test with pytest": [[20, "test-with-pytest"]], "Test with tox": [[20, "test-with-tox"]], "Troubleshooting": [[21, "troubleshooting"]], "Cannot load libomp": [[21, "cannot-load-libomp"]], "Issue of Initializing libomp": [[21, "issue-of-initializing-libomp"]], "Install in Virtual Environments": [[22, "install-in-virtual-environments"]], "Install in virtualenv Environment": [[22, "install-in-virtualenv-environment"]], "Install in conda Environment": [[22, "install-in-conda-environment"]]}, "indexentries": {"covariance_matrix() (in module detkit)": [[4, "detkit.covariance_matrix"]], "design_matrix() (in module detkit)": [[5, "detkit.design_matrix"]], "electrocardiogram() (in module detkit)": [[6, "detkit.electrocardiogram"]], "get_config() (in module detkit)": [[7, "detkit.get_config"]], "get_instructions_per_task() (in module detkit)": [[8, "detkit.get_instructions_per_task"]], "logdet() (in module detkit)": [[9, "detkit.logdet"]], "loggdet() (in module detkit)": [[10, "detkit.loggdet"]], "logpdet() (in module detkit)": [[11, "detkit.logpdet"]], "ortho_complement() (in module detkit)": [[12, "detkit.ortho_complement"]], "orthogonalize() (in module detkit)": [[13, "detkit.orthogonalize"]], "chunk_tasks": [[15, "term-CHUNK_TASKS"]], "count_perf": [[15, "term-COUNT_PERF"]], "cython_build_for_doc": [[15, "term-CYTHON_BUILD_FOR_DOC"]], "cython_build_in_source": [[15, "term-CYTHON_BUILD_IN_SOURCE"]], "debug_mode": [[15, "term-DEBUG_MODE"]], "long_int": [[15, "term-LONG_INT"]], "unsigned_long_int": [[15, "term-UNSIGNED_LONG_INT"]], "use_openmp": [[15, "term-USE_OPENMP"]], "use_symmetry": [[15, "term-USE_SYMMETRY"]], "for linux users:": [[16, "term-For-Linux-users"]], "for windows users:": [[16, "term-For-Windows-users"]], "for macos users:": [[16, "term-For-macOS-users"]]}})