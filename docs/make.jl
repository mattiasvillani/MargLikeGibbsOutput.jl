using Documenter, Literate, MargLikeGibbsOutput

ENV["GKSwstype"] = "100"  # Plots without a display

# Each script in examples/ becomes a page, with the code run by Documenter
const EXAMPLES = ["ridge", "nodal", "galaxy", "gnp"]
for example in EXAMPLES
    Literate.markdown(joinpath(@__DIR__, "..", "examples", "$example.jl"),
        joinpath(@__DIR__, "src", "examples"); credit = false)
end

makedocs(;
    sitename = "MargLikeGibbsOutput.jl",
    modules = [MargLikeGibbsOutput],
    authors = "Mattias Villani",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mattiasvillani.github.io/MargLikeGibbsOutput.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Method" => "method.md",
        "Examples" => ["examples/$example.md" for example in EXAMPLES],
        "API" => "api.md",
    ],
)

deploydocs(; repo = "github.com/mattiasvillani/MargLikeGibbsOutput.jl.git", devbranch = "main")
