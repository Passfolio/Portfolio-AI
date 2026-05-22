from __future__ import annotations

from pydantic import BaseModel


class UserInfo(BaseModel):
    id: int
    nickname: str
    githubLogin: str


class Career(BaseModel):
    careerKeywords: list[str] = []
    careerMajors: list[str] = []
    careerSkills: list[str] = []


class EducationHistory(BaseModel):
    universityDepartmentId: int | None = None
    name: str
    type: str = ""
    region: str = ""
    department: str = ""
    degree: str = ""
    duration: str = ""


class DevSpec(BaseModel):
    experience: int = 0
    careers: list[Career] = []
    educationHistory: list[EducationHistory] = []


class GithubProfile(BaseModel):
    login: str
    name: str | None = None
    avatarUrl: str | None = None


class GithubRepo(BaseModel):
    name: str
    description: str | None = None
    language: str | None = None
    htmlUrl: str
    fullName: str
    size: int = 0


class GenerateRequest(BaseModel):
    user: UserInfo
    devSpec: DevSpec
    githubProfile: GithubProfile
    githubRepos: list[GithubRepo] = []


class ArticleItem(BaseModel):
    title: str
    contents: str
    fileUrls: list[str] = []


class GenerateResponse(BaseModel):
    articles: list[ArticleItem]
