from molecule import Molecule

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

app = FastAPI()
molecules_db = {}


@app.get("/")
async def root():
    return FileResponse('public/index.html')


@app.get("/add")
async def add_molecule_get():
    return FileResponse('public/add.html')


@app.post("/add")
async def add_molecule_post(
        molecule_id: str = Form(...),
        name: str = Form(...),
        formula: str = Form(...),
        structure: str = Form(...),
        description: str = Form(...)
):
    global molecules_db

    molecule = Molecule(int(molecule_id), name, formula, structure, description)
    molecules_db[molecule_id] = molecule

    return RedirectResponse(url="/add", status_code=303)


@app.get("/get")
async def id_search_get():
    return FileResponse("public/id_search.html")


@app.post("/get")
async def id_search_post(molecule_id: str = Form(...)):
    result = list()
    try:
        result.append(molecules_db[molecule_id])
        return result

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Molecule with ID {molecule_id} not found"
        )


@app.get("/update")
async def update_get():
    return FileResponse("public/update.html")


@app.post("/update")
async def update_post(
        molecule_id: str = Form(...),
        name: str = Form(...),
        formula: str = Form(...),
        structure: str = Form(...),
        description: str = Form(...)
):
    global molecules_db

    try:
        molecules_db[molecule_id].name = name
        molecules_db[molecule_id].formula = formula
        molecules_db[molecule_id].smiles = structure
        molecules_db[molecule_id].description = description
        return RedirectResponse(url="/update", status_code=303)

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Molecule with ID {molecule_id} not found"
        )


@app.get("/list_all")
async def list_all_get():
    return FileResponse("public/list_all.html")


@app.put("/list_all")
async def list_all_put():
    global molecules_db
    return [{'id': molecule.mol_id, 'name': molecule.name, 'formula': molecule.formula,
             'smiles': molecule.smiles, 'description': molecule.description}
            for molecule in molecules_db.values()]


@app.get("/delete")
async def delete_get():
    return FileResponse("public/id_delete.html")


@app.delete("/delete/{molecule_id}")
async def delete_del(molecule_id: str):
    global molecules_db
    if molecule_id in molecules_db.keys():
        del molecules_db[molecule_id]
        return {"message": f"Molecule {molecule_id} deleted successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Molecule with ID {molecule_id} not found"
        )


@app.get("/substr_search")
async def substr_search_get():
    return FileResponse("public/substr_search.html")


@app.post("/substr_search")
async def substr_search_post():
    ans = {}
    list_of_smiles = [molecule.smiles for molecule in molecules_db.values()]
    its = 0
    for molecule in molecules_db.values():
        try:
            substructures = molecule.is_substructure(list(filter(lambda x: x != its, list_of_smiles)))
            its += 1
            ans[molecule.name] = substructures
        except AttributeError:
            raise HTTPException(
                status_code=404,
            )

    return ans
