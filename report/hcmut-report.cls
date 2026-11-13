\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-report}[2026/09/16 HCMUT report format]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\RequirePackage{iftex}
\ifPDFTeX
  \RequirePackage[utf8]{vntex,inputenc}
  \RequirePackage[T5]{fontenc}
\else
  \RequirePackage{fontspec}
  \setmainfont{Latin Modern Roman}
  \setsansfont{Latin Modern Sans}
  \setmonofont{Latin Modern Mono Light}
\fi
\RequirePackage[english]{babel}
\RequirePackage{anyfontsize,amsmath,amssymb}
\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{booktabs}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr}
\RequirePackage[super,square,sort]{natbib}
\RequirePackage{float}
\RequirePackage{xcolor}
\RequirePackage{listings}
\RequirePackage{hyperref}

\definecolor{codeblue}{RGB}{31,73,125}
\definecolor{codegray}{RGB}{242,242,242}
\definecolor{codegreen}{RGB}{0,128,0}
\definecolor{codered}{RGB}{180,0,0}
\definecolor{accent}{RGB}{46,117,182}

\lstdefinestyle{pythonstyle}{
  language=Python,
  backgroundcolor=\color{codegray},
  basicstyle=\ttfamily\footnotesize\color{codeblue},
  keywordstyle=\color{accent}\bfseries,
  commentstyle=\color{codegreen}\itshape,
  stringstyle=\color{codered},
  numberstyle=\tiny\color{gray},
  numbers=left,
  stepnumber=1,
  numbersep=8pt,
  frame=single,
  framerule=0.4pt,
  rulecolor=\color{gray!40},
  breaklines=true,
  breakatwhitespace=false,
  showstringspaces=false,
  tabsize=4,
  captionpos=b,
  xleftmargin=15pt
}
\lstset{style=pythonstyle}

\newcommand{\Month}{%
  \ifcase\month
  \or January\or February\or March\or April\or May\or June
  \or July\or August\or September\or October\or November\or December
  \fi
}

\newcommand{\@upperuniname}{Vietnam National University - Ho Chi Minh City}
\newcommand{\@uniname}{Ho Chi Minh City University of Technology}
\newcommand{\@deptname}{\textbf{Faculty of Computer Science and Engineering}}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}
\newcommand{\@coursename}{Course Name}
\newcommand{\coursename}[1]{\renewcommand{\@coursename}{#1}}
\newcommand{\@reporttype}{REPORT}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}
\title{Untitled}
\newcommand{\@advisor}{Advisor Name}
\newcommand{\advisor}[1]{\renewcommand{\@advisor}{#1}}
\newcommand{\@studygroup}{--}
\newcommand{\studygroup}[1]{\renewcommand{\@studygroup}{#1}}
\newcommand{\@stuname}{\@author}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}
\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}
\newcommand{\@reporttime}{\Month\ \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

\geometry{
  paper=a4paper,
  top=2.54cm,
  bottom=2.54cm,
  left=2.54cm,
  right=2.54cm,
  includeheadfoot=true,
  headheight=35pt
}

\onehalfspacing
\setlength{\parskip}{6pt}
\raggedbottom
\renewcommand{\arraystretch}{1.2}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
\AtBeginDocument{\counterwithin{equation}{section}}
\AtBeginDocument{\counterwithin{table}{section}}
\AtBeginDocument{\counterwithin{figure}{section}}
\AtBeginDocument{\renewcommand*\refname{References}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\pagestyle{fancy}
\fancyhf{}

% Dùng bản logo đã cắt viền trong suốt và quy định kích thước theo chiều cao.
% Cách này giữ logo ổn định, không phụ thuộc vào độ phân giải pixel của ảnh.
\newcommand{\hcmutlogofile}{graphics/hcmut-logo.png}
\newcommand{\hcmutlogo}[1][4.2cm]{%
  \IfFileExists{\hcmutlogofile}{%
    \includegraphics[height=#1,keepaspectratio]{\hcmutlogofile}%
  }{%
    \fbox{\rule{0pt}{3cm}\rule{3cm}{0pt}}%
  }%
}

\fancyhead[L]{%
  \bfseries
  \begin{tabular}{rl}
    \begin{picture}(28,18)
      \put(0,-7){%
        \IfFileExists{\hcmutlogofile}{%
          \includegraphics[height=25pt,keepaspectratio]{\hcmutlogofile}%
        }{}%
      }
    \end{picture}
    & \begin{tabular}{l}
      \ttfamily\@uniname \\
      \ttfamily\@deptname
    \end{tabular}
  \end{tabular}
}
\fancyfoot[C]{\thepage}

\newcommand{\coverpage}{%
  \clearpage
  \thispagestyle{empty}
  \begingroup
  \centering
  \vspace*{-3\baselineskip}
  {\MakeUppercase{\@upperuniname}\par}
  \medskip
  {\MakeUppercase{\@uniname}\par}
  \medskip
  {\MakeUppercase{\@deptname}\par}
  \vfill
  \hcmutlogo
  \vfill
  {\bfseries
  \begin{tabular}{c}
    \normalsize\@coursename \\[-1ex]
    \midrule
    \\[-1ex]
    \Large\@reporttype \\
    \\[1ex]
    \Large\parbox[c]{0.9\linewidth}{\centering \@title} \\[2ex]
  \end{tabular}\par}
  \vfill
  {\mdseries
    \begin{flushleft}
      \hspace*{1.5cm}\begin{tabular}{l l}
        \textbf{Giảng viên hướng dẫn:} & \@advisor \\
        \textbf{Nhóm học tập:} & \@studygroup \\
        \multicolumn{2}{l}{\textbf{Danh sách thực hiện:}} \\
      \end{tabular}
    \end{flushleft}
    \vspace{0.2cm}
    \centering
    \@stuname
  \par}
  \vfill
  {\MakeUppercase{\@reportplace, \@reporttime}\par}
  \endgroup
  \clearpage
  \pagestyle{fancy}
}