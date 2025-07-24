::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {#root}
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.a .b .c}
[Sitemap](/sitemap/sitemap.xml){.d}

::: {.e .f .g .h .i .j .k .l}
:::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.m .c}
::::::::::::::::::::::::: {.m .n .o .p .c}
::::: {.q .r .s .t .u .v .w .x .y .j .e .z .ab}
[Open in
app![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDEwIDEwIiBjbGFzcz0iZHQiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTS45ODUgOC40ODVhLjM3NS4zNzUgMCAxIDAgLjUzLjUzek04Ljc1IDEuMjVoLjM3NUEuMzc1LjM3NSAwIDAgMCA4Ljc1Ljg3NXpNOC4zNzUgNi41YS4zNzUuMzc1IDAgMSAwIC43NSAwek0zLjUuODc1YS4zNzUuMzc1IDAgMSAwIDAgLjc1em0tMS45ODUgOC4xNCA3LjUtNy41LS41My0uNTMtNy41IDcuNXptNi44Ni03Ljc2NVY2LjVoLjc1VjEuMjV6TTMuNSAxLjYyNWg1LjI1di0uNzVIMy41eiIgLz48L3N2Zz4=){.dt}](https://rsci.app.link/?%24canonical_url=https%3A%2F%2Fmedium.com%2Fp%2Fbc77bf74c31f&%7Efeature=LoOpenInAppButton&%7Echannel=ShowPostUnderUser&%7Estage=mobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------){.du
.ah .dv .bf .al .b .an .ao .ap .aq .ar .as .at .au .t .v .x .j .e .r .dw
.ab rel="noopener follow"}

:::: {.ac .r}
Sign up

::: {.fg .m}
[Sign
in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&source=post_page---top_nav_layout_nav-----------------------global_nav------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerSignInButton" rel="noopener follow"}
:::
::::
:::::

::::::::::::::::::::: {.q .r .s .ac .ae}
::::::: {.ac .r .af}
[![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI3MTkiIGhlaWdodD0iMTYwIiBmaWxsPSJub25lIiBhcmlhLWxhYmVsbGVkYnk9IndvcmRtYXJrLW1lZGl1bS1kZXNjIiB2aWV3Ym94PSIwIDAgNzE5IDE2MCIgY2xhc3M9ImF2IGF3IGF4Ij48ZGVzYyBpZD0id29yZG1hcmstbWVkaXVtLWRlc2MiPk1lZGl1bSBMb2dvPC9kZXNjPjxwYXRoIGZpbGw9IiMyNDI0MjQiIGQ9Im0xNzQuMTA0IDkuNzM0LjIxNS0uMDQ3VjguMDJIMTMwLjM5TDg5LjYgMTAzLjg5IDQ4LjgxIDguMDIxSDEuNDcydjEuNjY2bC4yMTIuMDQ3YzguMDE4IDEuODEgMTIuMDkgNC41MDkgMTIuMDkgMTQuMjQyVjEzNy45M2MwIDkuNzM0LTQuMDg3IDEyLjQzMy0xMi4xMDYgMTQuMjQzbC0uMjEyLjA0N3YxLjY3MWgzMi4xMTh2LTEuNjY1bC0uMjEzLS4wNDhjLTguMDE4LTEuODA5LTEyLjA4OS00LjUwOS0xMi4wODktMTQuMjQyVjMwLjU4Nmw1Mi4zOTkgMTIzLjMwNWgyLjk3Mmw1My45MjUtMTI2Ljc0M1YxNDAuNzVjLS42ODcgNy42ODgtNC43MjEgMTAuMDYyLTExLjk4MiAxMS43MDFsLS4yMTUuMDV2MS42NTJoNTUuOTQ4di0xLjY1MmwtLjIxNS0uMDVjLTcuMjY5LTEuNjM5LTExLjQtNC4wMTMtMTIuMDg3LTExLjcwMWwtLjAzNy0xMTYuNzc0aC4wMzdjMC05LjczMyA0LjA3MS0xMi40MzIgMTIuMDg3LTE0LjI0Mm0yNS41NTUgNzUuNDg4Yy45MTUtMjAuNDc0IDguMjY4LTM1LjI1MiAyMC42MDYtMzUuNTA3IDMuODA2LjA2MyA2Ljk5OCAxLjMxMiA5LjQ3OSAzLjcxNCA1LjI3MiA1LjExOCA3Ljc1MSAxNS44MTIgNy4zNjggMzEuNzkzem0tLjU1MyA1Ljc3aDY1LjU3M3YtLjI3NWMtLjE4Ni0xNS42NTYtNC43MjEtMjcuODM0LTEzLjQ2Ni0zNi4xOTYtNy41NTktNy4yMjctMTguNzUxLTExLjIwMy0zMC41MDctMTEuMjAzaC0uMjYzYy02LjEwMSAwLTEzLjU4NCAxLjQ4LTE4LjkwOSA0LjE2LTYuMDYxIDIuODA3LTExLjQwNyA3LjAwMy0xNS44NTUgMTIuNTExLTcuMTYxIDguODc0LTExLjQ5OSAyMC44NjYtMTIuNTU0IDM0LjM0M3EtLjA1LjYwNi0uMDkyIDEuMjEyYTUwIDUwIDAgMCAwLS4wNjUgMS4xNTEgODUuODA3IDg1LjgwNyAwIDAgMC0uMDk0IDUuNjg5Yy43MSAzMC41MjQgMTcuMTk4IDU0LjkxNyA0Ni40ODMgNTQuOTE3IDI1LjcwNSAwIDQwLjY3NS0xOC43OTEgNDQuNDA3LTQ0LjAxM2wtMS44ODYtLjY2NGMtNi41NTcgMTMuNTU2LTE4LjMzNCAyMS43NzEtMzEuNzM4IDIwLjc2OS0xOC4yOTctMS4zNjktMzIuMzE0LTE5LjkyMi0zMS4wNDItNDIuMzk1bTEzOS43MjIgNDEuMzU5Yy0yLjE1MSA1LjEwMS02LjYzOSA3LjkwOC0xMi42NTMgNy45MDhzLTExLjUxMy00LjEyOS0xNS40MTgtMTEuNjNjLTQuMTk3LTguMDUzLTYuNDA1LTE5LjQzNi02LjQwNS0zMi45MiAwLTI4LjA2NyA4LjcyOS00Ni4yMiAyMi4yNC00Ni4yMiA1LjY1NyAwIDEwLjExMSAyLjgwNyAxMi4yMzYgNy43MDR6bTQzLjQ5OSAyMC4wMDhjLTguMDE5LTEuODk3LTEyLjA4OS00LjcyMi0xMi4wODktMTQuOTUxVjEuMzA5bC00OC43MTYgMTQuMzUzdjEuNzU3bC4yOTktLjAyNGM2LjcyLS41NDMgMTEuMjc4LjM4NiAxMy45MjUgMi44MyAyLjA3MiAxLjkxNSAzLjA4MiA0Ljg1MyAzLjA4MiA4Ljk4N3YxOC42NmMtNC44MDMtMy4wNjctMTAuNTE2LTQuNTYtMTcuNDQ4LTQuNTYtMTQuMDU5IDAtMjYuOTA5IDUuOTItMzYuMTc2IDE2LjY3Mi05LjY2IDExLjIwNS0xNC43NjcgMjYuNTE4LTE0Ljc2NyA0NC4yNzgtLjAwMyAzMS43MiAxNS42MTIgNTMuMDM5IDM4Ljg1MSA1My4wMzkgMTMuNTk1IDAgMjQuNTMzLTcuNDQ5IDI5LjU0LTIwLjAxM3YxNi44NjVoNDMuNzExdi0xLjc0NnpNNDI0LjEgMTkuODE5YzAtOS45MDQtNy40NjgtMTcuMzc0LTE3LjM3NS0xNy4zNzQtOS44NTkgMC0xNy41NzMgNy42MzItMTcuNTczIDE3LjM3NHM3LjcyMSAxNy4zNzQgMTcuNTczIDE3LjM3NGM5LjkwNyAwIDE3LjM3NS03LjQ3IDE3LjM3NS0xNy4zNzRtMTEuNDk5IDEzMi41NDZjLTguMDE5LTEuODk3LTEyLjA4OS00LjcyMi0xMi4wODktMTQuOTUxaC0uMDM1VjQzLjYzNWwtNDMuNzE0IDEyLjU1MXYxLjcwNWwuMjYzLjAyNGM5LjQ1OC44NDIgMTIuMDQ3IDQuMSAxMi4wNDcgMTUuMTUydjgxLjA4Nmg0My43NTF2LTEuNzQ2em0xMTIuMDEzIDBjLTguMDE4LTEuODk3LTEyLjA4OS00LjcyMi0xMi4wODktMTQuOTUxVjQzLjYzNWwtNDEuNjIxIDEyLjEzN3YxLjcxbC4yNDYuMDI2YzcuNzMzLjgxMyA5Ljk2NyA0LjI1NyA5Ljk2NyAxNS4zNnY1OS4yNzljLTIuNTc4IDUuMTAyLTcuNDE1IDguMTMxLTEzLjI3NCA4LjMzNi05LjUwMyAwLTE0LjczNi02LjQxOS0xNC43MzYtMTguMDczVjQzLjYzOGwtNDMuNzE0IDEyLjU1djEuNzAzbC4yNjIuMDI0YzkuNDU5Ljg0IDEyLjA1IDQuMDk3IDEyLjA1IDE1LjE1MnY1MC4xN2E1Ni4zIDU2LjMgMCAwIDAgLjkxIDEwLjQ0NGwuNzg3IDMuNDIzYzMuNzAxIDEzLjI2MiAxMy4zOTggMjAuMTk3IDI4LjU5IDIwLjE5NyAxMi44NjggMCAyNC4xNDctNy45NjYgMjkuMTE1LTIwLjQzdjE3LjMxMWg0My43MTR2LTEuNzQ3em0xNjkuODE4IDEuNzg4di0xLjc0OWwtLjIxMy0uMDVjLTguNy0yLjAwNi0xMi4wODktNS43ODktMTIuMDg5LTEzLjQ5di02My43OWMwLTE5Ljg5LTExLjE3MS0zMS43NjEtMjkuODgzLTMxLjc2MS0xMy42NCAwLTI1LjE0MSA3Ljg4Mi0yOS41NjkgMjAuMTYtMy41MTctMTMuMDEtMTMuNjM5LTIwLjE2LTI4LjYwNi0yMC4xNi0xMy4xNDYgMC0yMy40NDkgNi45MzgtMjcuODY5IDE4LjY1N1Y0My42NDNMNTQ1LjQ4NyA1NS42OHYxLjcxNWwuMjYzLjAyNGM5LjM0NS44MjkgMTIuMDQ3IDQuMTgxIDEyLjA0NyAxNC45NXY4MS43ODRoNDAuNzg3di0xLjc0NmwtLjIxNS0uMDUzYy02Ljk0MS0xLjYzMS05LjE4MS00LjYwNi05LjE4MS0xMi4yMzlWNjYuOTk4YzEuODM2LTQuMjg5IDUuNTM3LTkuMzcgMTIuODUzLTkuMzcgOS4wODYgMCAxMy42OTIgNi4yOTYgMTMuNjkyIDE4LjY5N3Y3Ny44MjhoNDAuNzk3di0xLjc0NmwtLjIxNS0uMDUzYy02Ljk0LTEuNjMxLTkuMTgtNC42MDYtOS4xOC0xMi4yMzlWNzUuMDY2YTQyIDQyIDAgMCAwLS41NzgtNy4yNmMxLjk0Ny00LjY2MSA1Ljg2LTEwLjE3NyAxMy40NzUtMTAuMTc3IDkuMjE0IDAgMTMuNjkxIDYuMTE0IDEzLjY5MSAxOC42OTZ2NzcuODI4eiIgLz48L3N2Zz4=){.av
.aw
.ax}](https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------){.ag
.ah .ai .aj .ak .al .am .an .ao .ap .aq .ar .as .at .au .ac
aria-label="Homepage" testid="headerMediumLogo" rel="noopener follow"}

:::::: {.ay .i}
::::: {.ac .aj .az .ba .bb .r .bc .bd}
::: {.bm aria-hidden="false" aria-describedby="searchResults" aria-labelledby="searchResults"}
:::

::: {.bn .bo .ac}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTQuMDkyIDExLjA2YTYuOTUgNi45NSAwIDEgMSAxMy45IDAgNi45NSA2Ljk1IDAgMCAxLTEzLjkgMG02Ljk1LTguMDVhOC4wNSA4LjA1IDAgMSAwIDUuMTMgMTQuMjZsMy43NSAzLjc1YS41Ni41NiAwIDEgMCAuNzktLjc5bC0zLjczLTMuNzNBOC4wNSA4LjA1IDAgMCAwIDExLjA0MiAzeiIgY2xpcC1ydWxlPSJldmVub2RkIiAvPjwvc3ZnPg==)
:::
:::::
::::::
:::::::

:::::: {.i .l .x .fi .fj}
::::: {.fk .ac}
[](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerWriteButton" rel="noopener follow"}

:::: {.bf .b .bg .ab .du .fl .fm .ac .r .fn .fo}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0IiBhcmlhLWxhYmVsPSJXcml0ZSI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMTQgNGEuNS41IDAgMCAwIDAtMXptNyA2YS41LjUgMCAwIDAtMSAwem0tNy03SDR2MWgxMHpNMyA0djE2aDFWNHptMSAxN2gxNnYtMUg0em0xNy0xVjEwaC0xdjEwem0tMSAxYTEgMSAwIDAgMCAxLTFoLTF6TTMgMjBhMSAxIDAgMCAwIDEgMXYtMXpNNCAzYTEgMSAwIDAgMC0xIDFoMXoiIC8+PHBhdGggc3Ryb2tlPSJjdXJyZW50Q29sb3IiIGQ9Im0xNy41IDQuNS04LjQ1OCA4LjQ1OGEuMjUuMjUgMCAwIDAtLjA2LjA5OGwtLjgyNCAyLjQ3YS4yNS4yNSAwIDAgMCAuMzE2LjMxNmwyLjQ3LS44MjNhLjI1LjI1IDAgMCAwIC4wOTgtLjA2TDE5LjUgNi41bS0yLTIgMi4zMjMtMi4zMjNhLjI1LjI1IDAgMCAxIC4zNTQgMGwxLjY0NiAxLjY0NmEuMjUuMjUgMCAwIDEgMCAuMzU0TDE5LjUgNi41bS0yLTIgMiAyIiAvPjwvc3ZnPg==)

::: {.dt .m}
Write
:::
::::
:::::
::::::

::::: {.l .k .j .e}
:::: {.fk .ac}
[](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerSearchButton" rel="noopener follow"}

::: {.bf .b .bg .ab .du .fl .fm .ac .r .fn .fo}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0IiBhcmlhLWxhYmVsPSJTZWFyY2giPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNNC4wOTIgMTEuMDZhNi45NSA2Ljk1IDAgMSAxIDEzLjkgMCA2Ljk1IDYuOTUgMCAwIDEtMTMuOSAwbTYuOTUtOC4wNWE4LjA1IDguMDUgMCAxIDAgNS4xMyAxNC4yNmwzLjc1IDMuNzVhLjU2LjU2IDAgMSAwIC43OS0uNzlsLTMuNzMtMy43M0E4LjA1IDguMDUgMCAwIDAgMTEuMDQyIDN6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)
:::
::::
:::::

::::: {.fk .i .l .k}
:::: {.ac .r}
Sign up

::: {.fg .m}
[Sign
in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&source=post_page---top_nav_layout_nav-----------------------global_nav------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerSignInButton" rel="noopener follow"}
:::
::::
:::::

::::: {.m aria-hidden="false"}
:::: {.m .fl}
![](https://miro.medium.com/v2/resize:fill:64:64/1*dmbNkD5D-u45r44go_cf0g.png){.m
.fd .bx .by .bz .cx width="32" height="32" loading="lazy"
role="presentation"}

::: {.ft .bx .m .by .bz .fu .o .aj .fv}
:::
::::
:::::
:::::::::::::::::::::
:::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ac
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.ca .bh}
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: m
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.fw .fx .fy .fz .ga .m}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
:::
::::

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: m
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: m
[]{.m}

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: section
<div>

::: {.fu .gk .gl .gm .gn .go}
:::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.gp .gq .gr .gs .gt}
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.ac .cb}
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.ci .bh .gb .gc .gd .ge}
<div>

# **A/B Testing in Production MLOps: Why Traditional Deployments Fail ML Models** {#4f34 .pw-post-title .gu .gv .gw .bf .gx .gy .gz .ha .hb .hc .hd .he .hf .hg .hh .hi .hj .hk .hl .hm .hn .ho .hp .hq .hr .hs .ht .hu .hv .hw .bk testid="storyTitle"}

<div>

::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.speechify-ignore .ac .cp}
:::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.speechify-ignore .bh .m}
::::::::::::::::::::: {.ac .hx .hy .hz .ia .ib .ic .id .ie .if .ig .ih}
::::::::::::::::: {.ac .r .ih}
::::::::: {.ac .ii}
<div>

::::::: {.bm aria-hidden="false"}
:::::: {.be tabindex="-1"}
[](/?source=post_page---byline--bc77bf74c31f---------------------------------------){rel="noopener follow"
discover="true"}

::::: {.m .ij .ik .bx .il .im}
:::: {.m .fl}
![Jeffrey
Taylor](https://miro.medium.com/v2/resize:fill:64:64/1*dmbNkD5D-u45r44go_cf0g.png){.m
.fd .bx .by .bz .cx width="32" height="32" loading="lazy"
testid="authorPhoto"}

::: {.in .bx .m .by .bz .fu .o .io .fv}
:::
::::
:::::
::::::
:::::::

</div>
:::::::::

[]{.bf .b .bg .ab .bk}

::::::::: {.ip .ac .r}
:::::::: {.ac .r .iq}
:::::: {.ac .r}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
[[Jeffrey
Taylor](/?source=post_page---byline--bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .ir testid="authorName"
rel="noopener follow" discover="true"}]{.bf .b .bg .ab .bk}
:::
::::

</div>
::::::

::: {.is .bm}
:::
::::::::
:::::::::
:::::::::::::::::

::::: {.ac .r .it}
[]{.bf .b .bg .ab .du}

:::: {.ac .af}
[6 min read]{testid="storyReadTime"}

::: {.iu .iv .m aria-hidden="true"}
[[·]{.bf .b .bg .ab .du}]{.m aria-hidden="true"}
:::

[Jul 9, 2025]{testid="storyPublishDate"}
::::
:::::
:::::::::::::::::::::

:::::::::::::::::::::::::::::::::::: {.ac .cp .iw .ix .iy .iz .ja .jb .jc .jd .je .jf .jg .jh .ji .jj .jk .jl}
:::::::::::::: {.i .l .x .fi .fj .r}
:::::::::: {.kb .m}
::::::::: {.ac .r .kc .kd}
::::::: {.pw-multi-vote-icon .fl .ke .kf .kg .kh}
[](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fp%2Fbc77bf74c31f&operation=register&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&user=Jeffrey+Taylor&userId=4ca9cbd2ff28&source=---header_actions--bc77bf74c31f---------------------clap_footer------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerClapButton" rel="noopener follow"}

<div>

::::: {.bm aria-hidden="false"}
:::: {.be tabindex="-1"}
::: {.ki .ap .kj .kk .kl .km .an .kn .ko .kp .kh role="presentation"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld2JveD0iMCAwIDI0IDI0IiBhcmlhLWxhYmVsPSJjbGFwIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44MjggMTIgMy4yODJsLjYzLTIuNDU0ek0xMy45MTYgMy45NTNsMS41MjMtMi4xMTItMS4xODQtLjM5ek04LjU4OSAxLjg0bDEuNTIyIDIuMTEyLS4zMzctMi41MDF6TTE4LjUyMyAxOC45MmMtLjg2Ljg2LTEuNzUgMS4yNDYtMi42MiAxLjMzYTYgNiAwIDAgMCAuNDA3LS4zNzJjMi4zODgtMi4zODkgMi44Ni00Ljk1MSAxLjM5OS03LjYyM2wtLjkxMi0xLjYwMy0uNzktMS42NzJjLS4yNi0uNTYtLjE5NC0uOTguMjAzLTEuMjg4YS43LjcgMCAwIDEgLjU0Ni0uMTMyYy4yODMuMDQ2LjU0Ni4yMzEuNzI4LjVsMi4zNjMgNC4xNTdjLjk3NiAxLjYyNCAxLjE0MSA0LjIzNy0xLjMyNCA2LjcwMm0tMTAuOTk5LS40MzhMMy4zNyAxNC4zMjhhLjgyOC44MjggMCAwIDEgLjU4NS0xLjQwOC44My44MyAwIDAgMSAuNTg1LjI0MmwyLjE1OCAyLjE1N2EuMzY1LjM2NSAwIDAgMCAuNTE2LS41MTZsLTIuMTU3LTIuMTU4LTEuNDQ5LTEuNDQ5YS44MjYuODI2IDAgMCAxIDEuMTY3LTEuMTdsMy40MzggMy40NGEuMzYzLjM2MyAwIDAgMCAuNTE2IDAgLjM2NC4zNjQgMCAwIDAgMC0uNTE2TDUuMjkzIDkuNTEzbC0uOTctLjk3YS44MjYuODI2IDAgMCAxIDAtMS4xNjYuODQuODQgMCAwIDEgMS4xNjcgMGwuOTcuOTY4IDMuNDM3IDMuNDM2YS4zNi4zNiAwIDAgMCAuNTE3IDAgLjM2Ni4zNjYgMCAwIDAgMC0uNTE2TDYuOTc3IDcuODNhLjgyLjgyIDAgMCAxLS4yNDEtLjU4NC44Mi44MiAwIDAgMSAuODI0LS44MjZjLjIxOSAwIC40My4wODcuNTg0LjI0Mmw1Ljc4NyA1Ljc4N2EuMzY2LjM2NiAwIDAgMCAuNTg3LS40MTVsLTEuMTE3LTIuMzYzYy0uMjYtLjU2LS4xOTQtLjk4LjIwNC0xLjI4OWEuNy43IDAgMCAxIC41NDYtLjEzMmMuMjgzLjA0Ni41NDUuMjMyLjcyNy41MDFsMi4xOTMgMy44NmMxLjMwMiAyLjM4Ljg4MyA0LjU5LTEuMjc3IDYuNzUtMS4xNTYgMS4xNTYtMi42MDIgMS42MjctNC4xOSAxLjM2Ny0xLjQxOC0uMjM2LTIuODY2LTEuMDMzLTQuMDc5LTIuMjQ2TTEwLjc1IDUuOTcxbDIuMTIgMi4xMmMtLjQxLjUwMi0uNDY1IDEuMTctLjEyOCAxLjg5bC4yMi40NjUtMy41MjMtMy41MjNhLjguOCAwIDAgMS0uMDk3LS4zNjhjMC0uMjIuMDg2LS40MjguMjQxLS41ODRhLjg0Ny44NDcgMCAwIDEgMS4xNjcgMG03LjM1NSAxLjcwNWMtLjMxLS40NjEtLjc0Ni0uNzU4LTEuMjMtLjgzN2ExLjQ0IDEuNDQgMCAwIDAtMS4xMS4yNzVjLS4zMTIuMjQtLjUwNS41NDMtLjU5Ljg4MWExLjc0IDEuNzQgMCAwIDAtLjkwNi0uNDY1IDEuNDcgMS40NyAwIDAgMC0uODIuMTA2bC0yLjE4Mi0yLjE4MmExLjU2IDEuNTYgMCAwIDAtMi4yIDAgMS41NCAxLjU0IDAgMCAwLS4zOTYuNzAxIDEuNTYgMS41NiAwIDAgMC0yLjIxLS4wMSAxLjU1IDEuNTUgMCAwIDAtLjQxNi43NTNjLS42MjQtLjYyNC0xLjY0OS0uNjI0LTIuMjM3LS4wMzdhMS41NTcgMS41NTcgMCAwIDAgMCAyLjJjLS4yMzkuMS0uNTAxLjIzOC0uNzE1LjQ1M2ExLjU2IDEuNTYgMCAwIDAgMCAyLjJsLjUxNi41MTVhMS41NTYgMS41NTYgMCAwIDAtLjc1MyAyLjYxNUw3LjAxIDE5YzEuMzIgMS4zMTkgMi45MDkgMi4xODkgNC40NzUgMi40NDlxLjQ4Mi4wOC45NzEuMDhjLjg1IDAgMS42NTMtLjE5OCAyLjM5My0uNTc5LjIzMS4wMzMuNDYuMDU0LjY4Ni4wNTQgMS4yNjYgMCAyLjQ1Ny0uNTIgMy41MDUtMS41NjcgMi43NjMtMi43NjMgMi41NTItNS43MzQgMS40MzktNy41ODZ6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)
:::
::::
:::::

</div>
:::::::

::: {.pw-multi-vote-count .m .kq .kr .ks .kt .ku .kv .kw}
[\--]{.kx}
:::
:::::::::
::::::::::

<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld2JveD0iMCAwIDI0IDI0IiBjbGFzcz0ibGMiPjxwYXRoIGQ9Ik0xOC4wMDYgMTYuODAzYzEuNTMzLTEuNDU2IDIuMjM0LTMuMzI1IDIuMjM0LTUuMzIxQzIwLjI0IDcuMzU3IDE2LjcwOSA0IDEyLjE5MSA0UzQgNy4zNTcgNCAxMS40ODJjMCA0LjEyNiAzLjY3NCA3LjQ4MiA4LjE5MSA3LjQ4Mi44MTcgMCAxLjYyMi0uMTExIDIuMzkzLS4zMjcuMjMxLjIuNDguMzkxLjc0NC41NTkgMS4wNi42OTMgMi4yMDMgMS4wNDQgMy4zOTkgMS4wNDQuMjI0LS4wMDguNC0uMTEyLjQ4Ni0uMjg3YS40OS40OSAwIDAgMC0uMDQyLS41MThjLS40OTUtLjY3LS44NDUtMS4zNjQtMS4wNC0yLjA1N2E0IDQgMCAwIDEtLjEyNS0uNTk4em0tMy4xMjIgMS4wNTUtLjA2Ny0uMjIzLS4zMTUuMDk2YTggOCAwIDAgMS0yLjMxMS4zMzhjLTQuMDIzIDAtNy4yOTItMi45NTUtNy4yOTItNi41ODcgMC0zLjYzMyAzLjI2OS02LjU4OCA3LjI5Mi02LjU4OCA0LjAxNCAwIDcuMTEyIDIuOTU4IDcuMTEyIDYuNTkzIDAgMS43OTQtLjYwOCAzLjQ2OS0yLjAyNyA0LjcybC0uMTk1LjE2OHYuMjU1YzAgLjA1NiAwIC4xNTEuMDE2LjI5NS4wMjUuMjMxLjA4MS40NzguMTU0LjczMy4xNTQuNTU4LjM5OCAxLjExNy43MjIgMS42NTlhNS4zIDUuMyAwIDAgMS0yLjE2NS0uODQ1Yy0uMjc2LS4xNzYtLjcxNC0uMzgzLS45NDEtLjU5eiIgLz48L3N2Zz4=){.lc}
:::
::::

</div>
::::::::::::::

::::::::::::::::::::::: {.ac .r .jm .jn .jo .jp .jq .jr .js .jt .ju .jv .jw .jx .jy .jz .ka}
::: {.ld .l .k .j .e}
:::

:::::: {.i .l}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
[![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNSIgaGVpZ2h0PSIyNSIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI1IDI1IiBjbGFzcz0iZHUgbGUiIGFyaWEtbGFiZWw9IkFkZCB0byBsaXN0IGJvb2ttYXJrIGJ1dHRvbiI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMTggMi41YS41LjUgMCAwIDEgMSAwVjVoMi41YS41LjUgMCAwIDEgMCAxSDE5djIuNWEuNS41IDAgMSAxLTEgMFY2aC0yLjVhLjUuNSAwIDAgMSAwLTFIMTh6TTcgN2ExIDEgMCAwIDEgMS0xaDMuNWEuNS41IDAgMCAwIDAtMUg4YTIgMiAwIDAgMC0yIDJ2MTRhLjUuNSAwIDAgMCAuODA1LjM5NkwxMi41IDE3bDUuNjk1IDQuMzk2QS41LjUgMCAwIDAgMTkgMjF2LTguNWEuNS41IDAgMCAwLTEgMHY3LjQ4NWwtNS4xOTUtNC4wMTJhLjUuNSAwIDAgMC0uNjEgMEw3IDE5Ljk4NXoiIC8+PC9zdmc+){.du
.le}](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2Fbc77bf74c31f&operation=register&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&source=---header_actions--bc77bf74c31f---------------------bookmark_footer------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="headerBookmarkButton" rel="noopener follow"}
:::
::::

</div>
::::::

:::::::::::: {.fd .lf .cn}
::::::::::: {.m .af}
:::::::::: {.ac .cb}
::::::::: {.lg .lh .li .lj .lk .ll .ci .bh}
:::::::: ac
::::::: {.bm aria-hidden="false"}
<div>

::::: {.bm aria-hidden="false"}
:::: {.be tabindex="-1"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTMgMTJhOSA5IDAgMSAxIDE4IDAgOSA5IDAgMCAxLTE4IDBtOS0xMEM2LjQ3NyAyIDIgNi40NzcgMiAxMnM0LjQ3NyAxMCAxMCAxMCAxMC00LjQ3NyAxMC0xMFMxNy41MjMgMiAxMiAybTMuMzc2IDEwLjQxNi00LjU5OSAzLjA2NmEuNS41IDAgMCAxLS43NzctLjQxNlY4LjkzNGEuNS41IDAgMCAxIC43NzctLjQxNmw0LjU5OSAzLjA2NmEuNS41IDAgMCAxIDAgLjgzMiIgY2xpcC1ydWxlPSJldmVub2RkIiAvPjwvc3ZnPg==)

::: {.k .j .e}
Listen
:::
::::
:::::

</div>
:::::::
::::::::
:::::::::
::::::::::
:::::::::::
::::::::::::

::::::: {.bm aria-hidden="false" aria-describedby="postFooterSocialMenu" aria-labelledby="postFooterSocialMenu"}
<div>

::::: {.bm aria-hidden="false"}
:::: {.be tabindex="-1"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTE1LjIxOCA0LjkzMWEuNC40IDAgMCAxLS4xMTguMTMybC4wMTIuMDA2YS40NS40NSAwIDAgMS0uMjkyLjA3NC41LjUgMCAwIDEtLjMtLjEzbC0yLjAyLTIuMDJ2Ny4wN2MwIC4yOC0uMjMuNS0uNS41cy0uNS0uMjItLjUtLjV2LTcuMDRsLTIgMmEuNDUuNDUgMCAwIDEtLjU3LjA0aC0uMDJhLjQuNCAwIDAgMS0uMTYtLjMuNC40IDAgMCAxIC4xLS4zMmwyLjgtMi44YS41LjUgMCAwIDEgLjcgMGwyLjggMi43OWEuNDIuNDIgMCAwIDEgLjA2OC40OThtLS4xMDYuMTM4LjAwOC4wMDR2LS4wMXpNMTYgNy4wNjNoMS41YTIgMiAwIDAgMSAyIDJ2MTBhMiAyIDAgMCAxLTIgMmgtMTFjLTEuMSAwLTItLjktMi0ydi0xMGEyIDIgMCAwIDEgMi0ySDhhLjUuNSAwIDAgMSAuMzUuMTUuNS41IDAgMCAxIC4xNS4zNS41LjUgMCAwIDEtLjE1LjM1LjUuNSAwIDAgMS0uMzUuMTVINi40Yy0uNSAwLS45LjQtLjkuOXYxMC4yYS45LjkgMCAwIDAgLjkuOWgxMS4yYy41IDAgLjktLjQuOS0uOXYtMTAuMmMwLS41LS40LS45LS45LS45SDE2YS41LjUgMCAwIDEgMC0xIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)

::: {.k .j .e}
Share
:::
::::
:::::

</div>
:::::::
:::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

</div>

</div>

*Part 1 of 3: The Problem and Solution Framework*
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
# **About This Series** {#b8f3 .nl .nm .gw .bf .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .oa .ob .oc .od .oe .of .og .oh .oi .bk}

This 3-part series describes a fully operational, open-source
demonstration of an MLOps workflow for A/B testing financial models. The
entire system was built from the ground up to showcase production-ready
MLOps principles.

**The Complete Series:**

- [**Part 1**: Why A/B Testing ML Models is Different (This
  Article)]{#afe1}
- [[**Part 2**: Building Production A/B Testing
  Infrastructure](/building-production-a-b-testing-infrastructure-for-ml-models-75c8c3b36ba6){.ag
  .ow rel="noopener" discover="true"}]{#6550}
- [**Part 3**: Measuring Business Impact and ROI]{#4f25}
:::
::::
:::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
# **The Model Deployment Dilemma** {#185b .nl .nm .gw .bf .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .oa .ob .oc .od .oe .of .og .oh .oi .bk}

You've spent months training a new machine learning model. It shows
impressive accuracy in offline evaluation. Your stakeholders are
excited. But here's the million-dollar question: **How do you safely
deploy this model to production without risking your business?**

Traditional software deployment strategies fall short for ML models:

- [[**Blue-green
  deployments**](https://martinfowler.com/bliki/BlueGreenDeployment.html){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} are all-or-nothing:
  you risk everything on untested production behavior]{#08af}
- [[**Canary
  releases**](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#canary-deployments){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} help with
  infrastructure, but don't measure model-specific performance]{#46e8}
- [**Shadow testing** validates infrastructure but doesn't capture
  business impact]{#cd6d}

This is where **A/B testing for ML models** becomes essential.

# **Why A/B Testing is Different for ML Models** {#1be7 .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

Unlike traditional A/B testing (which focuses on UI changes and
conversion rates), ML A/B testing requires measuring:

<figure class="pf pg ph pi pj pk pc pd paragraph-image">
<div class="pl pm fl pn bh po" role="button" tabindex="0">
<div class="pc pd pe">

</div>
</div>
</figure>

**The key difference**: ML models have both *\*performance\** and
*\*business\** implications that must be measured simultaneously.

# **The Hidden Complexities of ML Model Deployment** {#aad7 .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

## **1. Performance vs. Business Impact Disconnect** {#5eaf .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

A model that performs better in offline evaluation might not deliver
better business results:

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
baseline_accuracy = 0.527    # 52.7%
advanced_accuracy = 0.852    # 85.2%
improvement = 0.325          # 32.5 percentage points

# But what happens in production?
covid_crash_accuracy = 0.571  # 57.1% during market stress
trading_return = -0.686       # -68.6% actual returns
transaction_costs = 0.019     # 1.9% per trade

# Reality check: 85.2% accuracy → -161% returns after costs
```

## **2. Model Behavior Changes in Production** {#e125 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Models behave differently in production due to:

- [**Data drift**: Production data differs from training data]{#fb3f}
- [**Concept drift**: The relationship between features and targets
  changes]{#17c9}
- [**Infrastructure differences**: Latency, memory constraints,
  concurrent load]{#8a50}
- [**Feedback loops**: Model predictions influence future data]{#99ce}

## **3. Risk Management Requirements** {#c507 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Financial models require special considerations

- [**Regulatory compliance**: Model decisions must be auditable]{#cf87}
- [**Risk tolerance**: Conservative approach needed for financial
  predictions]{#33b6}
- [**Fallback mechanisms**: Automatic reversion if model fails]{#28d9}
- [**Business continuity**: Zero-downtime deployment
  requirements]{#ab5b}

# **Our Real-World Example: Financial Forecasting** {#d868 .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

Let's demonstrate these challenges with a concrete example using a
financial forecasting platform built with:

- [[**Kubernetes**](https://kubernetes.io/docs/home/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} for orchestration]{#9dc5}
- [[**Seldon Core
  v2**](https://docs.seldon.io/projects/seldon-core/en/latest/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} for model serving and
  experiments]{#3664}
- [[**Prometheus**](https://prometheus.io/docs/introduction/overview/){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} for metrics
  collection]{#75e9}
- [[**Grafana**](https://grafana.com/docs/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} for visualization]{#b4ea}
- [[**Argo Workflows**](https://argoproj.github.io/argo-workflows/){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} for training
  pipelines]{#ae45}

<figure class="pf pg ph pi pj pk pc pd paragraph-image">
<div class="pl pm fl pn bh po" role="button" tabindex="0">
<div class="pc pd qo">

</div>
</div>
<figcaption><strong>Production MLOps A/B testing architecture with
GitOps automation</strong></figcaption>
</figure>

## **The Challenge** {#f3e0 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

We have two models:

- [**Baseline Model**: 52.7% accuracy, 45ms latency]{#11dc}
- [**Enhanced Model**: 85.2% accuracy, 62ms latency]{#a6ea}
- [**Critical Reality**: While the advanced model shows 85.2% accuracy
  in laboratory conditions, comprehensive backtesting revealed
  performance degradation during market stress (57.1% during COVID
  crash) and catastrophic losses (-68.6% to -161%) when transaction
  costs are included. **A/B testing would allow us to discover whether
  such failures occur in current live market conditions, while limiting
  exposure to 30% of capital.**]{#fd40}

# **The A/B Testing Solution Framework** {#1604 .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

## **1. Controlled Traffic Splitting** {#cb46 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Instead of all-or-nothing deployment, we split traffic:

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
# Seldon Core v2 Experiment Configuration
spec:
  default: baseline-predictor
  candidates:
    - name: baseline-predictor
      weight: 70
    - name: advanced-predictor
      weight: 30
  mirror:
    percent: 100
    name: traffic-mirror
```

**Key benefits:**

- [**70/30 split**: Conservative approach limits live exposure to 30% of
  capital]{#d994}
- [**Default fallback**: Automatic routing to baseline when live losses
  detected]{#b433}
- [**Traffic mirroring**: Copy live requests for offline
  analysis]{#1b75}
- [**Live validation**: Test whether backtest failures repeat in current
  market conditions]{#0708}

## **2. Comprehensive Metrics Collection** {#3679 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

We collect metrics that matter for ML models:

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
# Model-specific metrics
ab_test_model_accuracy{model_name="baseline-predictor"} 52.7
ab_test_model_accuracy{model_name="advanced-predictor"} 85.2

# Performance metrics
ab_test_response_time_seconds{model_name="baseline-predictor"} 0.045
ab_test_response_time_seconds{model_name="advanced-predictor"} 0.062

# Business impact metrics (live performance tracking)
ab_test_trading_return{model_name="advanced-predictor"} 2.3
ab_test_transaction_cost_impact{model_name="advanced-predictor"} -15.2
ab_test_requests_total{model_name="baseline-predictor"} 1851
ab_test_requests_total{model_name="advanced-predictor"} 649
```

## **3. Automated Decision Framework** {#5fb2 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
def make_deployment_decision(metrics):
    """Automated decision making based on comprehensive metrics"""
    trading_return = metrics['trading_return']
    transaction_cost_impact = metrics['transaction_cost_impact']
    
    # Live performance decision criteria
    if trading_return < -10.0:
        return "REJECT_AND_ROLLBACK"  # Live performance catastrophic
    elif transaction_cost_impact < -50.0:
        return "REJECT_AND_ROLLBACK"  # Live transaction costs too high
    elif trading_return > 5.0 and transaction_cost_impact > -10.0:
        return "RECOMMEND"  # Live performance good, increase traffic
    else:
        return "CONTINUE_TESTING"  # Need more live data
```

# **Key Principles for ML A/B Testing** {#44ab .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

## **1. Multi-Dimensional Success Criteria** {#67f6 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Traditional A/B testing focuses on a single metric (conversion rate). ML
A/B testing requires multiple success criteria:

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
success_criteria = {
    "primary": "live_trading_return > 5%",
    "secondary": "p95_latency < 200ms", 
    "guardrail": "live_transaction_cost_impact > -20%"
}
```

## **2. Conservative Traffic Allocation** {#41ef .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Unlike web A/B testing (often 50/50), ML models should use conservative
splits:

- [**Financial models**: 70/30 or 80/20]{#972b}
- [**Healthcare models**: 90/10 or 95/5]{#a212}
- [**Consumer models**: 60/40 or 70/30]{#d767}

## **3. Longer Test Duration** {#5d07 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

ML models need longer observation periods:

- [**Web A/B tests**: Hours to days]{#068a}
- [**ML A/B tests**: Days to weeks]{#edf2}
- [**Financial ML tests**: Weeks to months]{#6b90}

## **4. Backtest-Informed Live Testing** {#fbea .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
# Historical backtest insights inform live testing thresholds
backtest_lab_accuracy = 85.2
backtest_crisis_accuracy = 57.1  # COVID crash backtest
backtest_trading_return = -68.6  # Historical strategy returns

# Live A/B test success criteria based on backtest learnings
live_success_threshold = 5.0    # Must beat historical failures
live_rollback_threshold = -10.0  # Trigger based on backtest risks

# Transaction cost monitoring (live vs historical)
def monitor_live_vs_backtest():
    if live_trading_return < backtest_trading_return:
        trigger_rollback("Worse than historical worst case")
    elif live_trading_return > live_success_threshold:
        increase_traffic("Outperforming backtest expectations")
```

# **Common Pitfalls to Avoid** {#2d6b .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

## **1. Deploying Without Live Validation** {#cafa .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
# Dangerous approach
if lab_accuracy > baseline_accuracy:
    deploy_enhanced_model_to_100_percent()

# A/B testing approach
if lab_accuracy > baseline_accuracy:
    start_ab_test_with_30_percent_traffic()
    monitor_live_performance()
    if live_performance_meets_criteria():
        gradually_increase_traffic()
```

## **2. Not Accounting for Temporal Effects** {#b61a .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Models can perform differently across:

- [**Time of day**: Market hours vs. off-hours]{#4505}
- [**Day of week**: Weekdays vs. weekends]{#ed3c}
- [**Market conditions**: Bull vs. bear markets]{#7ddb}
- [**Seasonal patterns**: Holiday effects, earnings seasons]{#b299}

## **3. Insufficient Monitoring** {#5127 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

Critical alerts for ML A/B tests:

``` {.pf .pg .ph .pi .pj .qf .qg .qh .bp .qi .bb .bk}
# Model accuracy degradation
- alert: ModelAccuracyDegraded
  expr: ab_test_model_accuracy < 55
  for: 5m
  labels:
    severity: critical

# Trading return catastrophe
- alert: TradingReturnCatastrophe
  expr: ab_test_trading_return < -20
  for: 1m
  labels:
    severity: critical

# High response time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(ab_test_response_time_seconds_bucket[5m])) > 0.200
  for: 3m
  labels:
    severity: warning
```

# **The Path Forward** {#dd87 .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

A/B testing for ML models requires a fundamental shift in how we think
about model deployment:

- [1. **From binary to gradual**: Split traffic instead of
  all-or-nothing]{#7e49}
- [2. **From single to multi-metric**: Measure performance AND business
  impact]{#b067}
- [3. **From fast to patient**: Allow longer test durations]{#1942}
- [4. **From manual to automated**: Build decision frameworks]{#2163}
- [5. **From lab to reality**: Safely discover model failures under real
  market conditions]{#86b5}

# **What's Next** {#b86c .nl .nm .gw .bf .nn .no .ox .nq .nr .ns .oy .nu .nv .nw .oz .ny .nz .oa .pa .oc .od .oe .pb .og .oh .oi .bk}

In **Part 2** of this series, we'll dive deep into the technical
implementation:

- [Building production A/B testing infrastructure with Seldon Core
  v2]{#933f}
- [Implementing comprehensive metrics collection with Prometheus]{#629d}
- [Creating real-time dashboards with Grafana]{#780b}
- [Setting up automated alerting and rollback mechanisms]{#a92e}

In **Part 3**, we'll explore the business impact:

- [Measuring ROI of A/B testing infrastructure]{#1cc1}
- [Calculating business value of model improvements]{#db53}
- [Risk assessment and mitigation strategies]{#fd2f}
- [Building the business case for ML A/B testing]{#52ba}
:::
::::
:::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
# **Key Takeaways** {#b97e .nl .nm .gw .bf .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .oa .ob .oc .od .oe .of .og .oh .oi .bk}

1\. **Backtests reveal potential risks** --- Historical testing showed
85.2% lab accuracy degrading to catastrophic losses during crisis
periods

2\. **A/B testing validates live performance** --- Test whether backtest
failures repeat in current market conditions with limited exposure

3\. **Conservative traffic splits limit risk** --- 70/30 allocation caps
live losses while gathering performance data

4\. **Automated rollback prevents disasters** --- Real-time detection of
poor live performance triggers immediate fallback

5\. **Live validation complements backtesting** --- A/B testing bridges
the gap between historical analysis and current market reality
:::
::::
:::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
**Ready to build your own ML A/B testing system?** Continue with Part 2
where we'll implement the complete technical infrastructure.
:::
::::
:::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
# **Additional Resources** {#1bee .nl .nm .gw .bf .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .oa .ob .oc .od .oe .of .og .oh .oi .bk}

## **Essential Reading** {#94e6 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

- [[**MLOps
  Principles**](https://ml-ops.org/content/mlops-principles){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} --- Foundational concepts
  for ML in production]{#a05b}
- [[**Google's Rules of Machine
  Learning**](https://developers.google.com/machine-learning/guides/rules-of-ml){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} --- Best practices
  for ML engineering]{#5d14}
- [[**The Machine Learning Engineering
  Book**](https://www.mlebook.com/){.ag .ow rel="noopener ugc nofollow"
  target="_blank"} --- Comprehensive guide to production ML
  systems]{#97b4}

## **Tools and Frameworks** {#d1e9 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

- [[**Seldon Core**](https://docs.seldon.io/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} --- Advanced ML model
  serving and A/B testing]{#8e56}
- [[**MLflow**](https://mlflow.org/docs/latest/index.html){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} --- ML lifecycle
  management platform]{#e9c7}
- [[**Kubeflow**](https://www.kubeflow.org/docs/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} --- ML workflows on
  Kubernetes]{#99c8}

## **A/B Testing Resources** {#f2d6 .pq .nm .gw .bf .nn .pr .ps .dy .nr .pt .pu .ea .nv .mp .pv .pw .px .mt .py .pz .qa .mx .qb .qc .qd .qe .bk}

- [[**Optimizely's A/B Testing
  Guide**](https://www.optimizely.com/optimization-glossary/ab-testing/){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} --- Statistical
  fundamentals]{#4b4a}
- [[**Netflix Tech
  Blog**](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15){.ag
  .ow rel="noopener ugc nofollow" target="_blank"} --- Large-scale
  experimentation platform]{#db95}
- [[**Uber's Experimentation
  Platform**](https://eng.uber.com/experimentation-platform/){.ag .ow
  rel="noopener ugc nofollow" target="_blank"} --- Real-world ML A/B
  testing at scale]{#bd0b}
:::
::::
:::::

::: {.ac .cb .nd .ne .nf .ng role="separator"}
[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni .nj .nk}[]{.nh .bx .bm .ni
.nj}
:::

::::: {.gp .gq .gr .gs .gt}
:::: {.ac .cb}
::: {.ci .bh .gb .gc .gd .ge}
# Open Source Implementation {#6caf .nl .nm .gw .bf .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .oa .ob .oc .od .oe .of .og .oh .oi .bk}

*This is Part 1 of the "A/B Testing in Production MLOps" series. The
complete implementation is available as open source:*

- [**Platform**:
  [github.com/jtayl222/ml-platform](https://github.com/jtayl222/ml-platform){.ag
  .ow rel="noopener ugc nofollow" target="_blank"}]{#e708}
- [**Application**:
  [github.com/jtayl222/financial-mlops-pytorch](https://github.com/jtayl222/financial-mlops-pytorch){.ag
  .ow rel="noopener ugc nofollow" target="_blank"}]{#40bb}

*Follow me for more enterprise MLOps content and practical
implementation guides.*
:::
::::
:::::

</div>
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::: {.ac .cb}
:::::: {.ci .bh .gb .gc .gd .ge}
::::: {.qt .qu .ac .it}
:::: {.qv .ac}
[](https://medium.com/tag/seldon?source=post_page-----bc77bf74c31f---------------------------------------){.qw
.aj .an .ap rel="noopener follow"}

::: {.qx .fl .cx .qy .gg .qz .ra .bf .b .bg .ab .bk .rb}
Seldon
:::
::::
:::::
::::::
:::::::

::: m
:::

:::::::::::::::::::::::::::::::::::: {.m .af}
::::::::::::::::::::::::::::::::::: {.ac .cb}
:::::::::::::::::::::::::::::::::: {.ci .bh .gb .gc .gd .ge}
::::::::::::::::::::::::::::::::: {.ac .cp .rj}
:::::::::::::::::::::: {.ac .r .kc}
::::::::::::::::: {.rk .m}
[]{.m .rl .rm .rn .f .e}

::::::::: {.ac .r .kc .kd}
::::::: {.pw-multi-vote-icon .fl .ke .kf .kg .kh}
[](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fp%2Fbc77bf74c31f&operation=register&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&user=Jeffrey+Taylor&userId=4ca9cbd2ff28&source=---footer_actions--bc77bf74c31f---------------------clap_footer------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="footerClapButton" rel="noopener follow"}

<div>

::::: {.bm aria-hidden="false"}
:::: {.be tabindex="-1"}
::: {.ki .ap .kj .kk .kl .km .an .kn .ko .kp .kh role="presentation"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld2JveD0iMCAwIDI0IDI0IiBhcmlhLWxhYmVsPSJjbGFwIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44MjggMTIgMy4yODJsLjYzLTIuNDU0ek0xMy45MTYgMy45NTNsMS41MjMtMi4xMTItMS4xODQtLjM5ek04LjU4OSAxLjg0bDEuNTIyIDIuMTEyLS4zMzctMi41MDF6TTE4LjUyMyAxOC45MmMtLjg2Ljg2LTEuNzUgMS4yNDYtMi42MiAxLjMzYTYgNiAwIDAgMCAuNDA3LS4zNzJjMi4zODgtMi4zODkgMi44Ni00Ljk1MSAxLjM5OS03LjYyM2wtLjkxMi0xLjYwMy0uNzktMS42NzJjLS4yNi0uNTYtLjE5NC0uOTguMjAzLTEuMjg4YS43LjcgMCAwIDEgLjU0Ni0uMTMyYy4yODMuMDQ2LjU0Ni4yMzEuNzI4LjVsMi4zNjMgNC4xNTdjLjk3NiAxLjYyNCAxLjE0MSA0LjIzNy0xLjMyNCA2LjcwMm0tMTAuOTk5LS40MzhMMy4zNyAxNC4zMjhhLjgyOC44MjggMCAwIDEgLjU4NS0xLjQwOC44My44MyAwIDAgMSAuNTg1LjI0MmwyLjE1OCAyLjE1N2EuMzY1LjM2NSAwIDAgMCAuNTE2LS41MTZsLTIuMTU3LTIuMTU4LTEuNDQ5LTEuNDQ5YS44MjYuODI2IDAgMCAxIDEuMTY3LTEuMTdsMy40MzggMy40NGEuMzYzLjM2MyAwIDAgMCAuNTE2IDAgLjM2NC4zNjQgMCAwIDAgMC0uNTE2TDUuMjkzIDkuNTEzbC0uOTctLjk3YS44MjYuODI2IDAgMCAxIDAtMS4xNjYuODQuODQgMCAwIDEgMS4xNjcgMGwuOTcuOTY4IDMuNDM3IDMuNDM2YS4zNi4zNiAwIDAgMCAuNTE3IDAgLjM2Ni4zNjYgMCAwIDAgMC0uNTE2TDYuOTc3IDcuODNhLjgyLjgyIDAgMCAxLS4yNDEtLjU4NC44Mi44MiAwIDAgMSAuODI0LS44MjZjLjIxOSAwIC40My4wODcuNTg0LjI0Mmw1Ljc4NyA1Ljc4N2EuMzY2LjM2NiAwIDAgMCAuNTg3LS40MTVsLTEuMTE3LTIuMzYzYy0uMjYtLjU2LS4xOTQtLjk4LjIwNC0xLjI4OWEuNy43IDAgMCAxIC41NDYtLjEzMmMuMjgzLjA0Ni41NDUuMjMyLjcyNy41MDFsMi4xOTMgMy44NmMxLjMwMiAyLjM4Ljg4MyA0LjU5LTEuMjc3IDYuNzUtMS4xNTYgMS4xNTYtMi42MDIgMS42MjctNC4xOSAxLjM2Ny0xLjQxOC0uMjM2LTIuODY2LTEuMDMzLTQuMDc5LTIuMjQ2TTEwLjc1IDUuOTcxbDIuMTIgMi4xMmMtLjQxLjUwMi0uNDY1IDEuMTctLjEyOCAxLjg5bC4yMi40NjUtMy41MjMtMy41MjNhLjguOCAwIDAgMS0uMDk3LS4zNjhjMC0uMjIuMDg2LS40MjguMjQxLS41ODRhLjg0Ny44NDcgMCAwIDEgMS4xNjcgMG03LjM1NSAxLjcwNWMtLjMxLS40NjEtLjc0Ni0uNzU4LTEuMjMtLjgzN2ExLjQ0IDEuNDQgMCAwIDAtMS4xMS4yNzVjLS4zMTIuMjQtLjUwNS41NDMtLjU5Ljg4MWExLjc0IDEuNzQgMCAwIDAtLjkwNi0uNDY1IDEuNDcgMS40NyAwIDAgMC0uODIuMTA2bC0yLjE4Mi0yLjE4MmExLjU2IDEuNTYgMCAwIDAtMi4yIDAgMS41NCAxLjU0IDAgMCAwLS4zOTYuNzAxIDEuNTYgMS41NiAwIDAgMC0yLjIxLS4wMSAxLjU1IDEuNTUgMCAwIDAtLjQxNi43NTNjLS42MjQtLjYyNC0xLjY0OS0uNjI0LTIuMjM3LS4wMzdhMS41NTcgMS41NTcgMCAwIDAgMCAyLjJjLS4yMzkuMS0uNTAxLjIzOC0uNzE1LjQ1M2ExLjU2IDEuNTYgMCAwIDAgMCAyLjJsLjUxNi41MTVhMS41NTYgMS41NTYgMCAwIDAtLjc1MyAyLjYxNUw3LjAxIDE5YzEuMzIgMS4zMTkgMi45MDkgMi4xODkgNC40NzUgMi40NDlxLjQ4Mi4wOC45NzEuMDhjLjg1IDAgMS42NTMtLjE5OCAyLjM5My0uNTc5LjIzMS4wMzMuNDYuMDU0LjY4Ni4wNTQgMS4yNjYgMCAyLjQ1Ny0uNTIgMy41MDUtMS41NjcgMi43NjMtMi43NjMgMi41NTItNS43MzQgMS40MzktNy41ODZ6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)
:::
::::
:::::

</div>
:::::::

::: {.pw-multi-vote-count .m .kq .kr .ks .kt .ku .kv .kw}
[\--]{.kx}
:::
:::::::::

[]{.m .i .h .g .ro .rp}

::::::::: {.ac .r .kc .kd}
::::::: {.pw-multi-vote-icon .fl .ke .kf .kg .kh}
[](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fp%2Fbc77bf74c31f&operation=register&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&user=Jeffrey+Taylor&userId=4ca9cbd2ff28&source=---footer_actions--bc77bf74c31f---------------------clap_footer------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="footerClapButton" rel="noopener follow"}

<div>

::::: {.bm aria-hidden="false"}
:::: {.be tabindex="-1"}
::: {.ki .ap .kj .kk .kl .km .an .kn .ko .kp .kh role="presentation"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld2JveD0iMCAwIDI0IDI0IiBhcmlhLWxhYmVsPSJjbGFwIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44MjggMTIgMy4yODJsLjYzLTIuNDU0ek0xMy45MTYgMy45NTNsMS41MjMtMi4xMTItMS4xODQtLjM5ek04LjU4OSAxLjg0bDEuNTIyIDIuMTEyLS4zMzctMi41MDF6TTE4LjUyMyAxOC45MmMtLjg2Ljg2LTEuNzUgMS4yNDYtMi42MiAxLjMzYTYgNiAwIDAgMCAuNDA3LS4zNzJjMi4zODgtMi4zODkgMi44Ni00Ljk1MSAxLjM5OS03LjYyM2wtLjkxMi0xLjYwMy0uNzktMS42NzJjLS4yNi0uNTYtLjE5NC0uOTguMjAzLTEuMjg4YS43LjcgMCAwIDEgLjU0Ni0uMTMyYy4yODMuMDQ2LjU0Ni4yMzEuNzI4LjVsMi4zNjMgNC4xNTdjLjk3NiAxLjYyNCAxLjE0MSA0LjIzNy0xLjMyNCA2LjcwMm0tMTAuOTk5LS40MzhMMy4zNyAxNC4zMjhhLjgyOC44MjggMCAwIDEgLjU4NS0xLjQwOC44My44MyAwIDAgMSAuNTg1LjI0MmwyLjE1OCAyLjE1N2EuMzY1LjM2NSAwIDAgMCAuNTE2LS41MTZsLTIuMTU3LTIuMTU4LTEuNDQ5LTEuNDQ5YS44MjYuODI2IDAgMCAxIDEuMTY3LTEuMTdsMy40MzggMy40NGEuMzYzLjM2MyAwIDAgMCAuNTE2IDAgLjM2NC4zNjQgMCAwIDAgMC0uNTE2TDUuMjkzIDkuNTEzbC0uOTctLjk3YS44MjYuODI2IDAgMCAxIDAtMS4xNjYuODQuODQgMCAwIDEgMS4xNjcgMGwuOTcuOTY4IDMuNDM3IDMuNDM2YS4zNi4zNiAwIDAgMCAuNTE3IDAgLjM2Ni4zNjYgMCAwIDAgMC0uNTE2TDYuOTc3IDcuODNhLjgyLjgyIDAgMCAxLS4yNDEtLjU4NC44Mi44MiAwIDAgMSAuODI0LS44MjZjLjIxOSAwIC40My4wODcuNTg0LjI0Mmw1Ljc4NyA1Ljc4N2EuMzY2LjM2NiAwIDAgMCAuNTg3LS40MTVsLTEuMTE3LTIuMzYzYy0uMjYtLjU2LS4xOTQtLjk4LjIwNC0xLjI4OWEuNy43IDAgMCAxIC41NDYtLjEzMmMuMjgzLjA0Ni41NDUuMjMyLjcyNy41MDFsMi4xOTMgMy44NmMxLjMwMiAyLjM4Ljg4MyA0LjU5LTEuMjc3IDYuNzUtMS4xNTYgMS4xNTYtMi42MDIgMS42MjctNC4xOSAxLjM2Ny0xLjQxOC0uMjM2LTIuODY2LTEuMDMzLTQuMDc5LTIuMjQ2TTEwLjc1IDUuOTcxbDIuMTIgMi4xMmMtLjQxLjUwMi0uNDY1IDEuMTctLjEyOCAxLjg5bC4yMi40NjUtMy41MjMtMy41MjNhLjguOCAwIDAgMS0uMDk3LS4zNjhjMC0uMjIuMDg2LS40MjguMjQxLS41ODRhLjg0Ny44NDcgMCAwIDEgMS4xNjcgMG03LjM1NSAxLjcwNWMtLjMxLS40NjEtLjc0Ni0uNzU4LTEuMjMtLjgzN2ExLjQ0IDEuNDQgMCAwIDAtMS4xMS4yNzVjLS4zMTIuMjQtLjUwNS41NDMtLjU5Ljg4MWExLjc0IDEuNzQgMCAwIDAtLjkwNi0uNDY1IDEuNDcgMS40NyAwIDAgMC0uODIuMTA2bC0yLjE4Mi0yLjE4MmExLjU2IDEuNTYgMCAwIDAtMi4yIDAgMS41NCAxLjU0IDAgMCAwLS4zOTYuNzAxIDEuNTYgMS41NiAwIDAgMC0yLjIxLS4wMSAxLjU1IDEuNTUgMCAwIDAtLjQxNi43NTNjLS42MjQtLjYyNC0xLjY0OS0uNjI0LTIuMjM3LS4wMzdhMS41NTcgMS41NTcgMCAwIDAgMCAyLjJjLS4yMzkuMS0uNTAxLjIzOC0uNzE1LjQ1M2ExLjU2IDEuNTYgMCAwIDAgMCAyLjJsLjUxNi41MTVhMS41NTYgMS41NTYgMCAwIDAtLjc1MyAyLjYxNUw3LjAxIDE5YzEuMzIgMS4zMTkgMi45MDkgMi4xODkgNC40NzUgMi40NDlxLjQ4Mi4wOC45NzEuMDhjLjg1IDAgMS42NTMtLjE5OCAyLjM5My0uNTc5LjIzMS4wMzMuNDYuMDU0LjY4Ni4wNTQgMS4yNjYgMCAyLjQ1Ny0uNTIgMy41MDUtMS41NjcgMi43NjMtMi43NjMgMi41NTItNS43MzQgMS40MzktNy41ODZ6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)
:::
::::
:::::

</div>
:::::::

::: {.pw-multi-vote-count .m .kq .kr .ks .kt .ku .kv .kw}
[\--]{.kx}
:::
:::::::::
:::::::::::::::::

:::::: {.ay .ac}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld2JveD0iMCAwIDI0IDI0IiBjbGFzcz0ibGMiPjxwYXRoIGQ9Ik0xOC4wMDYgMTYuODAzYzEuNTMzLTEuNDU2IDIuMjM0LTMuMzI1IDIuMjM0LTUuMzIxQzIwLjI0IDcuMzU3IDE2LjcwOSA0IDEyLjE5MSA0UzQgNy4zNTcgNCAxMS40ODJjMCA0LjEyNiAzLjY3NCA3LjQ4MiA4LjE5MSA3LjQ4Mi44MTcgMCAxLjYyMi0uMTExIDIuMzkzLS4zMjcuMjMxLjIuNDguMzkxLjc0NC41NTkgMS4wNi42OTMgMi4yMDMgMS4wNDQgMy4zOTkgMS4wNDQuMjI0LS4wMDguNC0uMTEyLjQ4Ni0uMjg3YS40OS40OSAwIDAgMC0uMDQyLS41MThjLS40OTUtLjY3LS44NDUtMS4zNjQtMS4wNC0yLjA1N2E0IDQgMCAwIDEtLjEyNS0uNTk4em0tMy4xMjIgMS4wNTUtLjA2Ny0uMjIzLS4zMTUuMDk2YTggOCAwIDAgMS0yLjMxMS4zMzhjLTQuMDIzIDAtNy4yOTItMi45NTUtNy4yOTItNi41ODcgMC0zLjYzMyAzLjI2OS02LjU4OCA3LjI5Mi02LjU4OCA0LjAxNCAwIDcuMTEyIDIuOTU4IDcuMTEyIDYuNTkzIDAgMS43OTQtLjYwOCAzLjQ2OS0yLjAyNyA0LjcybC0uMTk1LjE2OHYuMjU1YzAgLjA1NiAwIC4xNTEuMDE2LjI5NS4wMjUuMjMxLjA4MS40NzguMTU0LjczMy4xNTQuNTU4LjM5OCAxLjExNy43MjIgMS42NTlhNS4zIDUuMyAwIDAgMS0yLjE2NS0uODQ1Yy0uMjc2LS4xNzYtLjcxNC0uMzgzLS45NDEtLjU5eiIgLz48L3N2Zz4=){.lc}
:::
::::

</div>
::::::
::::::::::::::::::::::

:::::::::::: {.ac .r}
:::::: {.nk .m .rq}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
[![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNSIgaGVpZ2h0PSIyNSIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI1IDI1IiBjbGFzcz0iZHUgbGUiIGFyaWEtbGFiZWw9IkFkZCB0byBsaXN0IGJvb2ttYXJrIGJ1dHRvbiI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMTggMi41YS41LjUgMCAwIDEgMSAwVjVoMi41YS41LjUgMCAwIDEgMCAxSDE5djIuNWEuNS41IDAgMSAxLTEgMFY2aC0yLjVhLjUuNSAwIDAgMSAwLTFIMTh6TTcgN2ExIDEgMCAwIDEgMS0xaDMuNWEuNS41IDAgMCAwIDAtMUg4YTIgMiAwIDAgMC0yIDJ2MTRhLjUuNSAwIDAgMCAuODA1LjM5NkwxMi41IDE3bDUuNjk1IDQuMzk2QS41LjUgMCAwIDAgMTkgMjF2LTguNWEuNS41IDAgMCAwLTEgMHY3LjQ4NWwtNS4xOTUtNC4wMTJhLjUuNSAwIDAgMC0uNjEgMEw3IDE5Ljk4NXoiIC8+PC9zdmc+){.du
.le}](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2Fbc77bf74c31f&operation=register&redirect=https%3A%2F%2Fjeftaylo.medium.com%2Fa-b-testing-in-production-mlops-why-traditional-deployments-fail-ml-models-bc77bf74c31f&source=---footer_actions--bc77bf74c31f---------------------bookmark_footer------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
testid="footerBookmarkButton" rel="noopener follow"}
:::
::::

</div>
::::::

::::::: {.nk .m .rq}
:::::: {.bm aria-hidden="false" aria-describedby="postFooterSocialMenu" aria-labelledby="postFooterSocialMenu"}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0ibm9uZSIgdmlld2JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTE1LjIxOCA0LjkzMWEuNC40IDAgMCAxLS4xMTguMTMybC4wMTIuMDA2YS40NS40NSAwIDAgMS0uMjkyLjA3NC41LjUgMCAwIDEtLjMtLjEzbC0yLjAyLTIuMDJ2Ny4wN2MwIC4yOC0uMjMuNS0uNS41cy0uNS0uMjItLjUtLjV2LTcuMDRsLTIgMmEuNDUuNDUgMCAwIDEtLjU3LjA0aC0uMDJhLjQuNCAwIDAgMS0uMTYtLjMuNC40IDAgMCAxIC4xLS4zMmwyLjgtMi44YS41LjUgMCAwIDEgLjcgMGwyLjggMi43OWEuNDIuNDIgMCAwIDEgLjA2OC40OThtLS4xMDYuMTM4LjAwOC4wMDR2LS4wMXpNMTYgNy4wNjNoMS41YTIgMiAwIDAgMSAyIDJ2MTBhMiAyIDAgMCAxLTIgMmgtMTFjLTEuMSAwLTItLjktMi0ydi0xMGEyIDIgMCAwIDEgMi0ySDhhLjUuNSAwIDAgMSAuMzUuMTUuNS41IDAgMCAxIC4xNS4zNS41LjUgMCAwIDEtLjE1LjM1LjUuNSAwIDAgMS0uMzUuMTVINi40Yy0uNSAwLS45LjQtLjkuOXYxMC4yYS45LjkgMCAwIDAgLjkuOWgxMS4yYy41IDAgLjktLjQuOS0uOXYtMTAuMmMwLS41LS40LS45LS45LS45SDE2YS41LjUgMCAwIDEgMC0xIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIC8+PC9zdmc+)
:::
::::

</div>
::::::
:::::::
::::::::::::
:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::: {.rr .m}
:::::::::::::::::::::: {.ac .cb}
::::::::::::::::::::: {.ci .bh .gb .gc .gd .ge}
:::::::::::::::::::: {.ac .ig .ie .ic .rs .rt}
::::::::::: {.ru .rv .rw .rx .ry .rz .sa .sb .sc .sd .ac .cp}
::::: {.i .l}
[](/?source=post_page---post_author_info--bc77bf74c31f---------------------------------------){tabindex="0"
rel="noopener follow" discover="true"}

:::: {.m .fl}
![Jeffrey
Taylor](https://miro.medium.com/v2/resize:fill:96:96/1*dmbNkD5D-u45r44go_cf0g.png){.m
.fd .bx .se .sf .cx width="48" height="48" loading="lazy"}

::: {.ft .bx .m .se .sf .fu .o .aj .sg}
:::
::::
:::::

::::: {.k .j .e}
[](/?source=post_page---post_author_info--bc77bf74c31f---------------------------------------){tabindex="0"
rel="noopener follow" discover="true"}

:::: {.m .fl}
![Jeffrey
Taylor](https://miro.medium.com/v2/resize:fill:128:128/1*dmbNkD5D-u45r44go_cf0g.png){.m
.fd .bx .sh .si .cx width="64" height="64" loading="lazy"}

::: {.ft .bx .m .sh .si .fu .o .aj .sg}
:::
::::
:::::

:::: {.k .j .e .sj .rq}
::: ac
:::
::::
:::::::::::

:::::::: {.ac .co .ca}
::::::: {.sk .sl .sm .sn .so .m}
[](/?source=post_page---post_author_info--bc77bf74c31f---------------------------------------){.ag
.ah .ai .ak .al .am .an .ao .ap .aq .ar .as .at .au .ac .r
rel="noopener follow" discover="true"}

## [Written by Jeffrey Taylor]{.gp .sp} {#written-by-jeffrey-taylor .pw-author-name .bf .sq .sr .ss .st .su .sv .sw .mp .pw .px .mt .pz .qa .mx .qc .qd .bk}

::::: {.qv .ac .ii}
::: {.m .rq}
[[28
followers](/followers?source=post_page---post_author_info--bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .ir
rel="noopener follow" discover="true"}]{.pw-follower-count .bf .b .bg
.ab .du}
:::

::: {.bf .b .bg .ab .du .ac .sx}
[[·]{.bf .b .bg .ab .du}]{.sy .m aria-hidden="true"}[14
following](/following?source=post_page---post_author_info--bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .ir
rel="noopener follow" discover="true"}
:::
:::::

::: {.sz .m}
:::
:::::::
::::::::

:::: {.i .l}
::: ac
:::
::::
::::::::::::::::::::
:::::::::::::::::::::
::::::::::::::::::::::
:::::::::::::::::::::::

:::::::::::: {.ta .m}
::: {.tb .bh .s .rr}
:::

:::::::::: {.ac .cb}
::::::::: {.ci .bh .gb .gc .gd .ge}
::::::: {.ac .r .cp}
## No responses yet {#no-responses-yet .bf .sq .no .nq .nr .ns .nu .nv .nw .ny .nz .oa .oc .od .oe .og .oh .bk}

:::::: {.ac .tc}
<div>

:::: {.bm aria-hidden="false"}
::: {.be tabindex="-1"}
[![](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNSIgaGVpZ2h0PSIyNSIgYXJpYS1sYWJlbD0iU2hpZWxkIHdpdGggYSBjaGVja21hcmsiIHZpZXdib3g9IjAgMCAyNSAyNSI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNMTEuOTg3IDUuMDM2YS43NTQuNzU0IDAgMCAxIC45MTQtLjAxYy45NzIuNzIxIDEuNzY3IDEuMjE4IDIuNiAxLjU0My44MjguMzIyIDEuNzE5LjQ4NSAyLjg4Ny41MDVhLjc1NS43NTUgMCAwIDEgLjc0MS43NTdjLS4wMTggMy42MjMtLjQzIDYuMjU2LTEuNDQ5IDguMjEtMS4wMzQgMS45ODQtMi42NjIgMy4yMDktNC45NjYgNC4wODNhLjc1Ljc1IDAgMCAxLS41MzctLjAwM2MtMi4yNDMtLjg3NC0zLjg1OC0yLjA5NS00Ljg5Ny00LjA3NC0xLjAyNC0xLjk1MS0xLjQ1Ny00LjU4My0xLjQ3Ni04LjIxNmEuNzU1Ljc1NSAwIDAgMSAuNzQxLS43NTdjMS4xOTUtLjAyIDIuMS0uMTgyIDIuOTIzLS41MDMuODI3LS4zMjIgMS42LS44MTUgMi41MTktMS41MzVtLjQ2OC45MDNjLS44OTcuNjktMS43MTcgMS4yMS0yLjYyMyAxLjU2NC0uODk4LjM1LTEuODU2LjUyNy0zLjAyNi41NjUuMDM3IDMuNDUuNDY5IDUuODE3IDEuMzYgNy41MTUuODg0IDEuNjg0IDIuMjUgMi43NjIgNC4yODQgMy41NzEgMi4wOTItLjgxIDMuNDY1LTEuODkgNC4zNDQtMy41NzUuODg2LTEuNjk4IDEuMjk5LTQuMDY1IDEuMzM0LTcuNTEyLTEuMTQ5LS4wMzktMi4wOTEtLjIxNy0yLjk5LS41NjctLjkwNi0uMzUzLTEuNzQ1LS44NzMtMi42ODMtMS41NjFtLS4wMDkgOS4xNTVhMi42NzIgMi42NzIgMCAxIDAgMC01LjM0NCAyLjY3MiAyLjY3MiAwIDAgMCAwIDUuMzQ0bTAgMWEzLjY3MiAzLjY3MiAwIDEgMCAwLTcuMzQ0IDMuNjcyIDMuNjcyIDAgMCAwIDAgNy4zNDRtLTEuODEzLTMuNzc3LjUyNS0uNTI2LjkxNi45MTcgMS42MjMtMS42MjUuNTI2LjUyNi0yLjE0OSAyLjE1MnoiIGNsaXAtcnVsZT0iZXZlbm9kZCIgLz48L3N2Zz4=)](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page---post_responses--bc77bf74c31f---------------------------------------){.td
.te rel="noopener follow" target="_blank"}
:::
::::

</div>
::::::
:::::::

::: {.tf .tg .th .ti .tj .m}
:::
:::::::::
::::::::::
::::::::::::

:::::::::::::::::: {.tk .tl .tm .tn .to .m .bw}
::::::::::::::::: {.i .l .k}
::: {.tb .bh .tp .tq}
:::

::::::::::::::: {.ac .cb}
:::::::::::::: {.ci .bh .gb .gc .gd .ge}
::::::::::::: {.tr .ac .kc .it}
::: {.ts .tt .m}
[](https://help.medium.com/hc/en-us?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Help
:::

::: {.ts .tt .m}
[](https://medium.statuspage.io/?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Status
:::

::: {.ts .tt .m}
[](https://medium.com/about?autoplay=1&source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

About
:::

::: {.ts .tt .m}
[](https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Careers
:::

::: {.ts .tt .m}
[](mailto:pressinquiries@medium.com){.ag .ah .ai .fh .ak .al .am .an .ao
.ap .aq .ar .as .at .au rel="noopener follow"}

Press
:::

::: {.ts .tt .m}
[](https://blog.medium.com/?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Blog
:::

::: {.ts .tt .m}
[](https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Privacy
:::

::: {.ts .tt .m}
[](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Rules
:::

::: {.ts .tt .m}
[](https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Terms
:::

::: {.ts .m}
[](https://speechify.com/medium?source=post_page-----bc77bf74c31f---------------------------------------){.ag
.ah .ai .fh .ak .al .am .an .ao .ap .aq .ar .as .at .au
rel="noopener follow"}

Text to speech
:::
:::::::::::::
::::::::::::::
:::::::::::::::
:::::::::::::::::
::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
