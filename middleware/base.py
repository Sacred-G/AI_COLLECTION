from pandasai.middlewares import ChartsMiddleware

class CustomChartsMiddleware(ChartsMiddleware):
    def run(self, code: str) -> str:
        processed = []
        for line in code.split("\n"):
            if line.find("plt.close()") != -1:
                idx = line.find("plt")
                blank = "".join([' ' for c in range(idx)])

                # Set font properties
                processed.append(blank + "plt.rcParams['font.sans-serif']=['Arial']")
                processed.append(blank + "plt.rcParams['axes.unicode_minus']=False")

                 #Uncomment the line below if you want to save the plot as 'temp_chart.png'
                processed.append(blank + "plt.savefig('temp_chart.png')")

                processed.append(line)
            else:
                processed.append(line)

        code = "\n".join(processed)
        return code
