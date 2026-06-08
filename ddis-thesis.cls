\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\def\filename{ddis-thesis.cls}
\ProvidesClass{ddis-thesis}[2006/06/06 v1.0e ddis LaTeX thesis class]

\LoadClass[a4paper, 12pt]{report}

\setlength {\marginparwidth }{3cm}
\setlength{\headheight}{15pt}

\RequirePackage{listings}
\RequirePackage{soul}
\RequirePackage{xcolor}
\RequirePackage{url}
\RequirePackage[pdftex]{hyperref}
\RequirePackage[ngerman,english]{babel}
\RequirePackage{amsmath, amsthm, amssymb}
\RequirePackage[utf8]{inputenc}
\RequirePackage[linecolor=gray,bordercolor=gray,backgroundcolor=yellow]{todonotes}
\RequirePackage{mathrsfs}
\RequirePackage{amsfonts}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{graphicx}
\RequirePackage{csquotes}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{helvet}
\RequirePackage[toc, numberedsection, acronym]{glossaries}
\RequirePackage[backend=biber, maxcitenames=2, maxbibnames=3, style=numeric, natbib=true]{biblatex}
\RequirePackage{cleveref}

\makeglossaries
\setglossarysection{section}
\renewcommand{\glossarysection}[2][]{}
\loadglsentries{resources/glossary}

% Set Table of Content Depth
\setcounter{tocdepth}{2}

\definecolor{darkblue}{RGB}{0,0,139}

\renewcommand{\chaptermark}[1]{%
    \markboth{\thechapter.\ #1}{}}


\hypersetup{
    colorlinks=true,
    allcolors=darkblue
}

\definecolor{gray}{gray}{0.5}


\newcommand*{\fontchapnumber}{\sffamily\fontsize{100}{130}\selectfont}

\titleformat{\chapter}[display]
{\sffamily\Huge\bfseries} % Text format
{\vspace{-1.5em}\color{gray}\fontchapnumber\thechapter} % Number format
{-20pt} % Spacing between number and title
{\sffamily} % Title format

\titleformat{\section}{\sffamily\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}



% Original name = Contents
\renewcommand{\contentsname}{Table of Contents}
% Original name = Bibliography
\renewcommand\bibname{References}

\newenvironment{abstracten}{%
    \chapter*{Abstract}
}{}

\newenvironment{abstractde}{%
    \chapter*{Zusammenfassung}
}{}

\newenvironment{acknowledgements}{%
    \chapter*{Acknowledgements}}{}

\newcommand{\declaration}{%
    \input{resources/declaration_en}
}

\newcommand{\ddistitlepage}{%
    \input{resources/title-page}
}

\graphicspath{ {figures/} }

\addto\extrasenglish{
    \renewcommand{\chapterautorefname}{Chapter}
    \renewcommand{\sectionautorefname}{Section}
    \renewcommand{\subsectionautorefname}{Section}
    \renewcommand{\subsubsectionautorefname}{Section}
    \renewcommand{\bibname}{References}
}

% Add fancy headers and footers
\pagestyle{fancy}
\fancyhf{} % Clear all header and footer fields
\fancyhead[L]{\thepage} % Left side of header: Chapter name
\fancyhead[C]{} % Center of header: empty
\fancyhead[R]{\leftmark} % Right side of header: page number


% *************** Bibliography ***************
\addbibresource{resources/sources.bib}


\DefineBibliographyStrings{english}{%
    andothers = {\textit{et al}\adddot}
}
\endinput
