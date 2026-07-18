//////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Innovation.cpp
// Description: Implementation for the Innovation and InnovationDatabase classes.
///////////////////////////////////////////////////////////////////////////////


#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Innovation.h"
#include "Genes.h"
#include "Genome.h"
#include "Assert.h"

namespace NEAT
{


// Creates an empty database
InnovationDatabase::InnovationDatabase()
{
    m_NextInnovationNum = 1; // innovations start at 1
    m_NextNeuronID = 1;      // neuron IDs start at 1
    m_Innovations.clear();
}

// Creates an empty database but this time sets the next innov number and neuron ID
InnovationDatabase::InnovationDatabase(int a_LastInnovationNum, int a_LastNeuronID)
{
    if (a_LastInnovationNum < 0 || a_LastNeuronID < 0)
    {
        throw std::invalid_argument(
            "Innovation counters cannot be negative");
    }

    m_NextInnovationNum = a_LastInnovationNum + 1;
    m_NextNeuronID = a_LastNeuronID + 1;
    m_Innovations.clear();
}



// Initializes an empty database
void InnovationDatabase::Init(int a_LastInnovationNum, int a_LastNeuronID)
{
    if (a_LastInnovationNum < 0 || a_LastNeuronID < 0)
    {
        throw std::invalid_argument(
            "Innovation counters cannot be negative");
    }
    Flush();

    m_NextNeuronID = a_LastNeuronID + 1;
    m_NextInnovationNum = a_LastInnovationNum + 1;
}

// Initializes a database from a given genome
void InnovationDatabase::Init(const Genome& a_Genome)
{
    m_Innovations.clear();
    for(unsigned int i=0; i<a_Genome.NumLinks(); i++)
    {
        Innovation t_innov( a_Genome.GetLinkByIndex(i).InnovationID(), NEW_LINK, a_Genome.GetLinkByIndex(i).FromNeuronID(), a_Genome.GetLinkByIndex(i).ToNeuronID(), NONE, -1);
        m_Innovations.emplace_back(t_innov);
    }

    m_NextNeuronID = a_Genome.GetLastNeuronID() + 1;
    m_NextInnovationNum = a_Genome.GetLastInnovationID() + 1;
}


void InnovationDatabase::Init(std::ifstream& a_DataFile)
{
    m_Innovations.clear();
    m_NextInnovationNum = 0;
    m_NextNeuronID = 0;

    std::string t_str;

    // search for InnovationDatabaseStart
    while (a_DataFile >> t_str && t_str != "InnovationDatabaseStart")
    {
    }
    if (t_str != "InnovationDatabaseStart")
        throw std::runtime_error(
            "InnovationDatabase::Init: missing start marker.");

    // Read the last innov numbers
    a_DataFile >> t_str;
    a_DataFile >> m_NextInnovationNum;
    a_DataFile >> t_str;
    a_DataFile >> m_NextNeuronID;
    if (!a_DataFile || m_NextInnovationNum < 1 || m_NextNeuronID < 1)
    {
        throw std::runtime_error(
            "InnovationDatabase::Init: invalid counters.");
    }


    // Read the database until InnovationDatabaseEnd is encountered
    while (a_DataFile >> t_str)
    {
        if (t_str == "Innovation")
        {
            // Read in the innovation
            int t_id, t_from, t_to, t_innovtype, t_neurontype, t_nid;

            a_DataFile >> t_id;
            a_DataFile >> t_innovtype;
            a_DataFile >> t_from;
            a_DataFile >> t_to;
            a_DataFile >> t_neurontype;
            a_DataFile >> t_nid;
            if (!a_DataFile ||
                (t_innovtype != NEW_NEURON &&
                 t_innovtype != NEW_LINK) ||
                t_id < 1 || t_from < 1 || t_to < 1)
            {
                throw std::runtime_error(
                    "InnovationDatabase::Init: invalid innovation.");
            }

            m_Innovations.emplace_back( Innovation(t_id, static_cast<InnovationType>(t_innovtype), t_from, t_to, static_cast<NeuronType>(t_neurontype), t_nid) );
        }
        else if (t_str == "InnovationDatabaseEnd")
        {
            return;
        }
    }
    throw std::runtime_error(
        "InnovationDatabase::Init: missing end marker.");
}


// The file is assumed to be opened
void InnovationDatabase::Save(FILE *a_file)
{
    if (a_file == nullptr)
        throw std::invalid_argument(
            "InnovationDatabase::Save: file is null.");
    fprintf(a_file, "InnovationDatabaseStart\n");
    fprintf(a_file, "NextInnovNum: %d\n", m_NextInnovationNum);
    fprintf(a_file, "NextNeuronID: %d\n", m_NextNeuronID);

    // Now save all innovations
    for(unsigned int i=0; i<m_Innovations.size(); i++)
    {
        fprintf(a_file, "Innovation %d %d %d %d %d %d\n", m_Innovations[i].ID(), static_cast<int>(m_Innovations[i].InnovType()), m_Innovations[i].FromNeuronID(), m_Innovations[i].ToNeuronID(), static_cast<int>(m_Innovations[i].GetNeuronType()), m_Innovations[i].NeuronID());
    }
    fprintf(a_file, "InnovationDatabaseEnd\n\n");
}

std::string InnovationDatabase::Serialize() const
{
    std::ostringstream output;
    output << "InnovationDatabaseStart\n";
    output << "NextInnovNum: " << m_NextInnovationNum << '\n';
    output << "NextNeuronID: " << m_NextNeuronID << '\n';
    for (const auto& innovation : m_Innovations)
    {
        output << "Innovation " << innovation.ID() << ' '
               << static_cast<int>(innovation.InnovType()) << ' '
               << innovation.FromNeuronID() << ' ' << innovation.ToNeuronID()
               << ' ' << static_cast<int>(innovation.GetNeuronType()) << ' '
               << innovation.NeuronID() << '\n';
    }
    output << "InnovationDatabaseEnd\n";
    return output.str();
}

InnovationDatabase InnovationDatabase::Deserialize(const std::string& data)
{
    std::istringstream input(data);
    std::string token;
    input >> token;
    if (token != "InnovationDatabaseStart")
        throw std::runtime_error(
            "InnovationDatabase::Deserialize: missing start marker.");

    InnovationDatabase database;
    input >> token >> database.m_NextInnovationNum;
    if (token != "NextInnovNum:")
        throw std::runtime_error(
            "InnovationDatabase::Deserialize: missing innovation counter.");
    input >> token >> database.m_NextNeuronID;
    if (token != "NextNeuronID:")
        throw std::runtime_error(
            "InnovationDatabase::Deserialize: missing neuron counter.");
    if (database.m_NextInnovationNum < 1 ||
        database.m_NextNeuronID < 1)
        throw std::runtime_error(
            "InnovationDatabase::Deserialize: invalid counters.");

    database.m_Innovations.clear();
    while (input >> token)
    {
        if (token == "InnovationDatabaseEnd")
            return database;
        if (token != "Innovation")
            throw std::runtime_error(
                "InnovationDatabase::Deserialize: malformed innovation.");
        int id, type, from, to, neuron_type, neuron_id;
        input >> id >> type >> from >> to >> neuron_type >> neuron_id;
        if (!input || (type != NEW_NEURON && type != NEW_LINK) ||
            id < 1 || from < 1 || to < 1)
            throw std::runtime_error(
                "InnovationDatabase::Deserialize: incomplete innovation.");
        database.m_Innovations.emplace_back(
            id, static_cast<InnovationType>(type), from, to,
            static_cast<NeuronType>(neuron_type), neuron_id);
    }
    throw std::runtime_error(
        "InnovationDatabase::Deserialize: missing end marker.");
}



// Checks the database if the innovation has already occured
// Returns the innovation id if true or -1 if false
// If it is a NEW_LINK innovation, in & out specify the neuron IDs being connected
// If it is a NEW_NEURON innovation, in & out specify the connection that was split
int InnovationDatabase::CheckInnovation(int a_In, int a_Out, InnovationType a_Type) const
{
    if (a_In <= 0 || a_Out <= 0 ||
        (a_Type != NEW_NEURON && a_Type != NEW_LINK))
        throw std::invalid_argument("Invalid innovation query");

    // search the list for a match
    for(unsigned int i=0; i < m_Innovations.size(); i++)
    {
        if ((m_Innovations[i].FromNeuronID() == a_In) && (m_Innovations[i].ToNeuronID() == a_Out) && (m_Innovations[i].InnovType() == a_Type))
        {
            // match found?
            return m_Innovations[i].ID();
        }
    }

    // not found
    return -1;
}


int InnovationDatabase::CheckLastInnovation(int a_In, int a_Out, InnovationType a_Type) const
{
    if (a_In <= 0 || a_Out <= 0 ||
        (a_Type != NEW_NEURON && a_Type != NEW_LINK))
        throw std::invalid_argument("Invalid innovation query");
    int t_ID = -1;

    // search the list for a match
    for(unsigned int i=0; i < m_Innovations.size(); i++)
    {
        if ((m_Innovations[i].FromNeuronID() == a_In) && (m_Innovations[i].ToNeuronID() == a_Out) && (m_Innovations[i].InnovType() == a_Type))
        {
            // match found?
            t_ID = m_Innovations[i].ID();
        }
    }

    return t_ID;
}


// returns a list of indexes in the database of identical innovations
std::vector<int> InnovationDatabase::CheckAllInnovations(int a_In, int a_Out, InnovationType a_Type) const
{
    if (a_In <= 0 || a_Out <= 0 ||
        (a_Type != NEW_NEURON && a_Type != NEW_LINK))
        throw std::invalid_argument("Invalid innovation query");

    std::vector<int> t_idxs;
    t_idxs.clear();

    // search the list for a match
    for(unsigned int i=0; i < m_Innovations.size(); i++)
    {
        if ((m_Innovations[i].FromNeuronID() == a_In) && (m_Innovations[i].ToNeuronID() == a_Out) && (m_Innovations[i].InnovType() == a_Type))
        {
            // match found?
            t_idxs.emplace_back( i );
        }
    }

    return t_idxs;
}



// Returns the neuron ID given the in and out neurons
// If not found, returns -1
int InnovationDatabase::FindNeuronID(int a_In, int a_Out) const
{
    if (a_In <= 0 || a_Out <= 0)
        throw std::invalid_argument("Invalid neuron innovation query");

    // search the list for a match
    for(unsigned int i=0; i < m_Innovations.size(); i++)
    {
        if ((m_Innovations[i].FromNeuronID() == a_In) && (m_Innovations[i].ToNeuronID() == a_Out) && (m_Innovations[i].InnovType() == NEW_NEURON))
        {
            // match found?
            return m_Innovations[i].NeuronID();
        }
    }

    // Not found
    return -1;
}

int InnovationDatabase::FindLastNeuronID(int a_In, int a_Out) const
{
    if (a_In <= 0 || a_Out <= 0)
        throw std::invalid_argument("Invalid neuron innovation query");
    int t_ID = -1;

    // search the list for a match
    for(unsigned int i=0; i < m_Innovations.size(); i++)
    {
        if ((m_Innovations[i].FromNeuronID() == a_In) && (m_Innovations[i].ToNeuronID() == a_Out) && (m_Innovations[i].InnovType() == NEW_NEURON))
        {
            // match found?
            t_ID = m_Innovations[i].NeuronID();
        }
    }

    return t_ID;
}


// Adds a new link innovation and returns its ID
// Increments the m_NextInnovationNum internally
int InnovationDatabase::AddLinkInnovation(int a_In, int a_Out)
{
    if (a_In <= 0 || a_Out <= 0)
        throw std::invalid_argument("Innovation endpoints must be positive");

    m_Innovations.emplace_back( Innovation(m_NextInnovationNum, NEW_LINK, a_In, a_Out, NONE, -1) );
    m_NextInnovationNum++;

    return (m_NextInnovationNum - 1);
}




// Adds a new neuron innovation and returns the new neuron ID
// in and out specify the connection that was split
// type specifies the type of neuron
// Increments the m_NextNeuronID and m_NextInnovationNum internally
int InnovationDatabase::AddNeuronInnovation(int a_In, int a_Out, NeuronType a_NType)
{
    if (a_In <= 0 || a_Out <= 0 || a_NType != HIDDEN)
        throw std::invalid_argument(
            "Neuron innovations require positive endpoints and a hidden neuron");

    m_Innovations.emplace_back( Innovation(m_NextInnovationNum, NEW_NEURON, a_In, a_Out, a_NType, m_NextNeuronID) );
    m_NextInnovationNum++;
    m_NextNeuronID++;

    return (m_NextNeuronID - 1);
}




// Clears all innovations in the database
void InnovationDatabase::Flush()
{
    m_Innovations.clear();
}




} // namespace NEAT
